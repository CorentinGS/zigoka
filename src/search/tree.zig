const std = @import("std");
const model_layout = @import("model_layout");
const search_config = @import("search_config");
const buffers = @import("../infer/buffers.zig");
const network = @import("../infer/network.zig");
const rules = @import("../rules/rules.zig");

pub const R = rules.Rules(model_layout.CONFIG.board_size);

const config = model_layout.CONFIG;

comptime {
    if (config.input_planes != rules.INPUT_PLANES) @compileError("the model does not read Moka's 12 input planes");
    if (config.policy_moves != R.MOVES) @compileError("the policy does not cover every point plus pass");
}

pub const NODES: u32 = search_config.nodes;
pub const EDGES: u32 = search_config.edges;

const EXPLORATION: f64 = 1.5;
const FIRST_PLAY_REDUCTION: f64 = 0.2;
const NONE: u32 = std.math.maxInt(u32);
const MAX_DEPTH: u32 = R.MAX_MOVE_COUNT;
const REUSE_DEPTH: u32 = 2;

pub const Status = enum(u32) {
    ok = 0,
    weights_not_loaded = 1,
    bad_batch_count = 2,
    non_finite_output = 3,
    game_over = 4,
    pool_full = 5,
    not_started = 6,
};

const Kind = enum(u8) { pending, terminal, expanded };

const Step = struct { node: u32, edge: u32 };

const Leaf = struct { node: u32, depth: u32, path: [MAX_DEPTH]Step };

const Outcome = enum { leaf, visited, collision, pool_full };

var positions: [NODES]R.Position = undefined;
var node_kind: [NODES]Kind = undefined;
var node_value: [NODES]f64 = undefined;
var node_visits: [NODES]u32 = undefined;
var node_in_flight: [NODES]u32 = undefined;
var edge_start: [NODES]u32 = undefined;
var edge_count: [NODES]u16 = undefined;

var edge_move: [EDGES]u16 = undefined;
var edge_prior: [EDGES]f64 = undefined;
var edge_visits: [EDGES]u32 = undefined;
var edge_in_flight: [EDGES]u32 = undefined;
var edge_value_sum: [EDGES]f64 = undefined;
var edge_child: [EDGES]u32 = undefined;

var remap: [NODES]u32 = undefined;

var leaves: [buffers.MAX_BATCH]Leaf = undefined;
var leaf_analyses: [buffers.MAX_BATCH]R.Analysis = undefined;

var nodes_used: u32 = 0;
var edges_used: u32 = 0;
var edges_reserved: u32 = 0;
var started = false;
var reused = false;

pub const STATIC_BYTES: u64 = @sizeOf(@TypeOf(positions)) + @sizeOf(@TypeOf(node_kind)) +
    @sizeOf(@TypeOf(node_value)) + @sizeOf(@TypeOf(node_visits)) + @sizeOf(@TypeOf(node_in_flight)) +
    @sizeOf(@TypeOf(edge_start)) + @sizeOf(@TypeOf(edge_count)) + @sizeOf(@TypeOf(edge_move)) +
    @sizeOf(@TypeOf(edge_prior)) + @sizeOf(@TypeOf(edge_visits)) + @sizeOf(@TypeOf(edge_in_flight)) +
    @sizeOf(@TypeOf(edge_value_sum)) + @sizeOf(@TypeOf(edge_child)) + @sizeOf(@TypeOf(remap)) +
    @sizeOf(@TypeOf(leaves)) + @sizeOf(@TypeOf(leaf_analyses));

fn features(slot: usize) *[R.FEATURES]f32 {
    return buffers.input[slot * buffers.INPUT_STRIDE ..][0..R.FEATURES];
}

/// The score the search gives a finished game, from Black's side. At 9×9, Moka's final score. At
/// 19×19, only the pass-dead stones are removed: Moka's capture aftermath also takes every chain left
/// in atari, so the search passed instead of playing captures that the final score then counted.
pub fn searchScore(position: *const R.Position) i32 {
    if (R.SIZE != 19) return R.finalScore(position);
    const area = R.passAliveArea(&position.board);
    var board = position.board;
    for (&board, area) |*stone, owner| {
        if (stone.* != rules.EMPTY and owner == -stone.*) stone.* = rules.EMPTY;
    }
    return R.areaScore(&board);
}

/// Sign of the final area score, for the player to move.
fn finalValue(position: *const R.Position) f64 {
    const black_result: f64 = std.math.sign(@as(f64, @floatFromInt(searchScore(position))));
    return if (position.next_color == rules.BLACK) black_result else -black_result;
}

fn allocateNode(position: R.Position) u32 {
    const node = nodes_used;
    nodes_used += 1;
    positions[node] = position;
    node_visits[node] = 0;
    node_in_flight[node] = 0;
    edge_count[node] = 0;
    if (R.isGameOver(&positions[node])) {
        node_kind[node] = .terminal;
        node_value[node] = finalValue(&positions[node]);
    } else {
        node_kind[node] = .pending;
        edges_reserved += R.MOVES;
    }
    return node;
}

fn prepareLeaf(slot: usize, node: u32) void {
    leaf_analyses[slot] = R.analyze(&positions[node].board);
    R.encode(&positions[node], &leaf_analyses[slot], features(slot));
}

/// Same softmax as match/search.ts, in f64 over the candidate moves only.
fn expand(slot: usize, node: u32) void {
    const policy = buffers.policy_out[slot * config.policy_moves ..][0..config.policy_moves];
    var moves: [R.MOVES]u16 = undefined;
    const count = R.candidateMoves(&positions[node], &leaf_analyses[slot], &moves);
    const first = edges_used;
    edges_used += count;
    edges_reserved -= R.MOVES;
    var maximum = -std.math.inf(f64);
    for (moves[0..count]) |move| maximum = @max(maximum, @as(f64, policy[move]));
    var sum: f64 = 0;
    for (moves[0..count], first..) |move, edge| {
        edge_prior[edge] = @exp(@as(f64, policy[move]) - maximum);
        sum += edge_prior[edge];
    }
    for (moves[0..count], first..) |move, edge| {
        edge_prior[edge] /= sum;
        edge_move[edge] = move;
        edge_visits[edge] = 0;
        edge_in_flight[edge] = 0;
        edge_value_sum[edge] = 0;
        edge_child[edge] = NONE;
    }
    node_value[node] = buffers.value_out[slot];
    edge_start[node] = first;
    edge_count[node] = count;
    node_kind[node] = .expanded;
}

/// PUCT with first-play urgency; in-flight visits count as losses so that a batch spreads out.
fn select(node: u32) u32 {
    const total: f64 = @floatFromInt(node_visits[node] + node_in_flight[node]);
    const exploration = EXPLORATION * @sqrt(total + 1);
    const first_play_value = node_value[node] - FIRST_PLAY_REDUCTION;
    const first = edge_start[node];
    var selected = first;
    var selected_score = -std.math.inf(f64);
    for (first..first + edge_count[node]) |edge| {
        const in_flight: f64 = @floatFromInt(edge_in_flight[edge]);
        const visits = @as(f64, @floatFromInt(edge_visits[edge])) + in_flight;
        const mean_value = if (visits > 0) (edge_value_sum[edge] - in_flight) / visits else first_play_value;
        const score = mean_value + (exploration * edge_prior[edge]) / (1 + visits);
        if (score > selected_score) {
            selected = @intCast(edge);
            selected_score = score;
        }
    }
    return selected;
}

fn backup(path: []const Step, leaf_value: f64) void {
    var value = leaf_value;
    var step = path.len;
    while (step > 0) {
        step -= 1;
        value = -value;
        edge_visits[path[step].edge] += 1;
        edge_value_sum[path[step].edge] += value;
        node_visits[path[step].node] += 1;
    }
}

fn setInFlight(path: []const Step, add: bool) void {
    for (path) |step| {
        if (add) {
            edge_in_flight[step.edge] += 1;
            node_in_flight[step.node] += 1;
        } else {
            edge_in_flight[step.edge] -= 1;
            node_in_flight[step.node] -= 1;
        }
    }
}

fn hasRoom() bool {
    return nodes_used < NODES and edges_used + edges_reserved + R.MOVES <= EDGES;
}

fn descend(slot: usize) Outcome {
    const leaf = &leaves[slot];
    var node: u32 = 0;
    var depth: u32 = 0;
    while (true) {
        switch (node_kind[node]) {
            .terminal => {
                backup(leaf.path[0..depth], node_value[node]);
                return .visited;
            },
            .pending => return .collision,
            .expanded => {},
        }
        const edge = select(node);
        leaf.path[depth] = .{ .node = node, .edge = edge };
        depth += 1;
        if (edge_child[edge] != NONE) {
            node = edge_child[edge];
            continue;
        }
        if (!hasRoom()) return .pool_full;
        var position = positions[node];
        std.debug.assert(R.play(&position, edge_move[edge]));
        const child = allocateNode(position);
        edge_child[edge] = child;
        if (node_kind[child] == .terminal) {
            backup(leaf.path[0..depth], node_value[child]);
            return .visited;
        }
        leaf.node = child;
        leaf.depth = depth;
        setInFlight(leaf.path[0..depth], true);
        prepareLeaf(slot, child);
        return .leaf;
    }
}

fn evaluate(count: usize) Status {
    var status: Status = .ok;
    for (0..count) |slot| {
        if (!network.forward(slot)) status = .non_finite_output;
    }
    if (status != .ok) {
        started = false;
        return status;
    }
    for (leaves[0..count], 0..) |*leaf, slot| {
        expand(slot, leaf.node);
        setInFlight(leaf.path[0..leaf.depth], false);
        backup(leaf.path[0..leaf.depth], node_value[leaf.node]);
    }
    return .ok;
}

fn isSamePosition(left: *const R.Position, right: *const R.Position) bool {
    return left.ko == right.ko and left.move_count == right.move_count and
        left.last_move == right.last_move and left.previous_move == right.previous_move and
        left.pass_count == right.pass_count and left.next_color == right.next_color and
        std.mem.eql(i8, &left.board, &right.board);
}

/// Finds the node of `position` within `REUSE_DEPTH` moves of `node`.
fn findNode(node: u32, position: *const R.Position, depth: u32) ?u32 {
    if (isSamePosition(&positions[node], position)) return node;
    if (depth == REUSE_DEPTH or node_kind[node] != .expanded) return null;
    const first = edge_start[node];
    for (first..first + edge_count[node]) |edge| {
        if (edge_child[edge] == NONE) continue;
        if (findNode(edge_child[edge], position, depth + 1)) |found| return found;
    }
    return null;
}

/// Keeps only the subtree of `root`, moved to the front of the pools with `root` as node 0.
/// A child is always allocated after its parent and expanded in allocation order, so nodes and
/// edge ranges both move to lower indices and one ascending pass can move them in place.
fn keepSubtree(root: u32) void {
    @memset(remap[0..nodes_used], NONE);
    remap[root] = 0;
    var kept: u32 = 0;
    for (root..nodes_used) |node| {
        if (remap[node] == NONE) continue;
        remap[node] = kept;
        kept += 1;
        if (node_kind[node] != .expanded) continue;
        const first = edge_start[node];
        for (first..first + edge_count[node]) |edge| {
            if (edge_child[edge] != NONE) remap[edge_child[edge]] = 0;
        }
    }
    var edges_kept: u32 = 0;
    for (root..nodes_used) |node| {
        const target = remap[node];
        if (target == NONE) continue;
        positions[target] = positions[node];
        node_kind[target] = node_kind[node];
        node_value[target] = node_value[node];
        node_visits[target] = node_visits[node];
        node_in_flight[target] = 0;
        edge_count[target] = edge_count[node];
        if (node_kind[node] != .expanded) continue;
        const first = edge_start[node];
        std.debug.assert(edges_kept <= first);
        edge_start[target] = edges_kept;
        for (first..first + edge_count[node]) |edge| {
            edge_move[edges_kept] = edge_move[edge];
            edge_prior[edges_kept] = edge_prior[edge];
            edge_visits[edges_kept] = edge_visits[edge];
            edge_in_flight[edges_kept] = 0;
            edge_value_sum[edges_kept] = edge_value_sum[edge];
            edge_child[edges_kept] = if (edge_child[edge] == NONE) NONE else remap[edge_child[edge]];
            edges_kept += 1;
        }
    }
    nodes_used = kept;
    edges_used = edges_kept;
    edges_reserved = 0;
}

/// Starts a search from `root`. With `reuse`, keeps the subtree of the last search whose root is
/// `root`, if `root` is within `REUSE_DEPTH` moves of the last root; otherwise clears the tree and
/// evaluates `root`.
pub fn start(root: *const R.Position, reuse: bool) Status {
    reused = false;
    if (R.isGameOver(root)) {
        started = false;
        return .game_over;
    }
    if (reuse and started) {
        if (findNode(0, root, 0)) |node| {
            if (node_kind[node] == .expanded) {
                keepSubtree(node);
                reused = true;
                return .ok;
            }
        }
    }
    started = false;
    nodes_used = 0;
    edges_used = 0;
    edges_reserved = 0;
    const node = allocateNode(root.*);
    prepareLeaf(0, node);
    if (!network.forward(0)) return .non_finite_output;
    expand(0, node);
    started = true;
    return .ok;
}

/// Simulates until the root has `target` visits, evaluating up to `batch` leaves per network call.
pub fn run(target: u32, batch: u32) Status {
    if (!started) return .not_started;
    if (batch < 1 or batch > buffers.MAX_BATCH) return .bad_batch_count;
    while (node_visits[0] < target) {
        var count: usize = 0;
        var outcome: Outcome = .visited;
        while (count < batch and node_visits[0] + node_in_flight[0] < target) {
            outcome = descend(count);
            switch (outcome) {
                .leaf => count += 1,
                .visited => {},
                .collision, .pool_full => break,
            }
        }
        if (count > 0) {
            const status = evaluate(count);
            if (status != .ok) return status;
        }
        if (outcome == .pool_full) return .pool_full;
    }
    return .ok;
}

/// Whether the last `start` kept a subtree instead of evaluating the root.
pub fn startReused() bool {
    return reused;
}

pub fn rootVisits() u32 {
    return if (started) node_visits[0] else 0;
}

pub fn nodeCount() u32 {
    return nodes_used;
}

/// Most visited root move, ties broken by the higher prior.
pub fn bestMove() u32 {
    if (!started) return R.PASS;
    const first = edge_start[0];
    var best = first;
    for (first + 1..first + edge_count[0]) |edge| {
        if (edge_visits[edge] > edge_visits[best] or
            (edge_visits[edge] == edge_visits[best] and edge_prior[edge] > edge_prior[best]))
        {
            best = @intCast(edge);
        }
    }
    return edge_move[best];
}

pub fn rootChildCount() u32 {
    return if (started) edge_count[0] else 0;
}

pub fn rootChildMove(index: u32) u32 {
    return edge_move[edge_start[0] + index];
}

pub fn rootChildVisits(index: u32) u32 {
    return edge_visits[edge_start[0] + index];
}
