const std = @import("std");
const memory_config = @import("memory_config");
const buffers = @import("infer/buffers.zig");
const weights = @import("infer/weights.zig");
const tree = @import("search/tree.zig");

const R = tree.R;

pub const panic = std.debug.no_panic;

comptime {
    _ = @import("infer/exports.zig");
}

const Status = tree.Status;

var position: R.Position = .{};
var moves: [R.MOVES]u16 = undefined;

const MEMORY_BYTES: u64 = @as(u64, memory_config.memory_pages) * 64 * 1024;
const USED_BYTES: u64 = buffers.STATIC_BYTES + tree.STATIC_BYTES + memory_config.stack_bytes;

comptime {
    if (USED_BYTES * 5 > MEMORY_BYTES * 4) {
        @compileError(std.fmt.comptimePrint(
            "{d} pages ({d} bytes) leave less than 25% headroom over {d} bytes of buffers, tree and stack",
            .{ memory_config.memory_pages, MEMORY_BYTES, USED_BYTES },
        ));
    }
}

export fn position_ptr() *R.Position {
    return &position;
}

export fn moves_ptr() [*]u16 {
    return &moves;
}

export fn rules_play(move: u32) u32 {
    if (move > R.PASS) return 0;
    return @intFromBool(R.play(&position, @intCast(move)));
}

export fn rules_legal_moves() u32 {
    const analysis = R.analyze(&position.board);
    return R.legalMoves(&position, &analysis, &moves);
}

export fn rules_encode() void {
    const analysis = R.analyze(&position.board);
    R.encode(&position, &analysis, buffers.input[0..R.FEATURES]);
}

export fn rules_is_game_over() u32 {
    return @intFromBool(R.isGameOver(&position));
}

export fn rules_area_score() i32 {
    return R.areaScore(&position.board);
}

/// Writes the automatically dead stones in ascending order and returns how many there are.
export fn rules_dead_stones() u32 {
    const dead = R.deadStones(&position);
    var count: u32 = 0;
    var points = dead.iterator(.{});
    while (points.next()) |point| {
        moves[count] = @intCast(point);
        count += 1;
    }
    return count;
}

/// Starts a search from the position; `reuse` != 0 keeps the matching subtree of the last search.
export fn search_start(reuse: u32) Status {
    if (!weights.loaded) return .weights_not_loaded;
    return tree.start(&position, reuse != 0);
}

export fn search_reused() u32 {
    return @intFromBool(tree.startReused());
}

export fn search_run(target_visits: u32, batch: u32) Status {
    return tree.run(target_visits, batch);
}

export fn search_visits() u32 {
    return tree.rootVisits();
}

export fn search_nodes() u32 {
    return tree.nodeCount();
}

export fn search_node_capacity() u32 {
    return tree.NODES;
}

export fn search_best_move() u32 {
    return tree.bestMove();
}

export fn search_child_count() u32 {
    return tree.rootChildCount();
}

export fn search_child_move(index: u32) u32 {
    return tree.rootChildMove(index);
}

export fn search_child_visits(index: u32) u32 {
    return tree.rootChildVisits(index);
}
