const std = @import("std");

pub const BLACK: i8 = 1;
pub const WHITE: i8 = -1;
pub const EMPTY: i8 = 0;
pub const NO_MOVE: i16 = -1;
pub const KOMI: i32 = 7;
pub const INPUT_PLANES: u32 = 12;

const KOMI_NORMALIZATION: f64 = 15;
const GAME_OVER_PASS_COUNT = 2;
const PASS_ALIVE_MAXIMUM_INTERNAL_POINTS = 1;
const PASS_ALIVE_REQUIRED_VITAL_REGIONS = 2;

const Plane = enum(u32) {
    current = 0,
    opponent = 1,
    current_one_liberty = 2,
    opponent_one_liberty = 3,
    current_two_liberties = 4,
    opponent_two_liberties = 5,
    ko = 6,
    last_move = 7,
    previous_move = 8,
    last_pass = 9,
    previous_pass = 10,
    komi = 11,
};

/// Moka's rules (vendor/moka/src/game.ts) on an N×N board, without allocation.
pub fn Rules(comptime N: u16) type {
    return struct {
        pub const SIZE: u16 = N;
        pub const AREA: u16 = N * N;
        pub const PASS: u16 = AREA;
        pub const MOVES: u16 = AREA + 1;
        pub const FEATURES: u32 = @as(u32, AREA) * INPUT_PLANES;
        pub const MAX_MOVE_COUNT: u16 = switch (N) {
            9 => 120,
            19 => 540,
            else => @compileError("Moka defines move limits for 9×9 and 19×19 only"),
        };
        pub const MIN_PASS_MOVE_COUNT: u16 = switch (N) {
            9 => 20,
            19 => 89,
            else => @compileError("Moka defines move limits for 9×9 and 19×19 only"),
        };

        pub const Points = std.bit_set.Static(AREA);

        /// Read and written by JS at fixed offsets: the scalars come first so that they do not move with N.
        pub const Position = extern struct {
            ko: i16 = NO_MOVE,
            move_count: u16 = 0,
            last_move: i16 = NO_MOVE,
            previous_move: i16 = NO_MOVE,
            pass_count: u8 = 0,
            next_color: i8 = BLACK,
            board: [AREA]i8 = @splat(EMPTY),
        };

        /// Group of every stone, keyed by its lowest point, with liberty and stone counts per key.
        pub const Analysis = struct {
            key: [AREA]u16,
            liberties: [AREA]u16,
            stones: [AREA]u16,
        };

        const Adjacent = struct { points: [4]u16, len: u8 };

        const ADJACENT: [AREA]Adjacent = adjacent: {
            @setEvalBranchQuota(@as(u32, AREA) * 64);
            var table: [AREA]Adjacent = undefined;
            for (0..AREA) |point| {
                const row = point / N;
                const column = point % N;
                var entry: Adjacent = .{ .points = undefined, .len = 0 };
                if (row > 0) append(&entry, point - N);
                if (row < N - 1) append(&entry, point + N);
                if (column > 0) append(&entry, point - 1);
                if (column < N - 1) append(&entry, point + 1);
                table[point] = entry;
            }
            break :adjacent table;
        };

        fn append(entry: *Adjacent, point: usize) void {
            entry.points[entry.len] = @intCast(point);
            entry.len += 1;
        }

        pub fn adjacent(point: usize) []const u16 {
            return ADJACENT[point].points[0..ADJACENT[point].len];
        }

        const Group = struct { stones: Points, liberties: Points };

        fn findGroup(board: *const [AREA]i8, start: u16) Group {
            const color = board[start];
            var group: Group = .{ .stones = .empty, .liberties = .empty };
            var pending: [AREA]u16 = undefined;
            var pending_len: usize = 1;
            pending[0] = start;
            group.stones.set(start);
            while (pending_len > 0) {
                pending_len -= 1;
                for (adjacent(pending[pending_len])) |next| {
                    if (board[next] == EMPTY) {
                        group.liberties.set(next);
                    } else if (board[next] == color and !group.stones.isSet(next)) {
                        group.stones.set(next);
                        pending[pending_len] = next;
                        pending_len += 1;
                    }
                }
            }
            return group;
        }

        pub fn analyze(board: *const [AREA]i8) Analysis {
            var analysis: Analysis = .{ .key = @splat(0), .liberties = @splat(0), .stones = @splat(0) };
            var seen: Points = .empty;
            for (0..AREA) |point| {
                if (board[point] == EMPTY or seen.isSet(point)) continue;
                const group = findGroup(board, @intCast(point));
                seen.setUnion(group.stones);
                analysis.liberties[point] = @intCast(group.liberties.count());
                analysis.stones[point] = @intCast(group.stones.count());
                var stones = group.stones.iterator(.{});
                while (stones.next()) |stone| analysis.key[stone] = @intCast(point);
            }
            return analysis;
        }

        fn recordMove(position: *Position, move: u16) void {
            position.previous_move = position.last_move;
            position.last_move = @intCast(move);
            position.move_count += 1;
            position.next_color = -position.next_color;
        }

        /// Plays `move` (PASS included) and returns false, leaving `position` unchanged, when Moka rejects it.
        pub fn play(position: *Position, move: u16) bool {
            if (move == PASS) {
                position.pass_count +|= 1;
                position.ko = NO_MOVE;
                recordMove(position, move);
                return true;
            }
            if (move > PASS or position.board[move] != EMPTY or position.ko == move) return false;
            const color = position.next_color;
            var board = position.board;
            board[move] = color;
            var captured_count: u32 = 0;
            var captured_point: u16 = 0;
            for (adjacent(move)) |next| {
                if (board[next] != -color) continue;
                const group = findGroup(&board, next);
                if (group.liberties.count() > 0) continue;
                var stones = group.stones.iterator(.{});
                while (stones.next()) |stone| {
                    board[stone] = EMPTY;
                    captured_point = @intCast(stone);
                    captured_count += 1;
                }
            }
            const played = findGroup(&board, move);
            const liberty_count = played.liberties.count();
            if (liberty_count == 0) return false;
            position.board = board;
            position.pass_count = 0;
            position.ko = if (captured_count == 1 and played.stones.count() == 1 and liberty_count == 1)
                @intCast(captured_point)
            else
                NO_MOVE;
            recordMove(position, move);
            return true;
        }

        /// Same answer as Moka's playMove on an empty point, from the liberty counts of the adjacent groups.
        pub fn isLegal(position: *const Position, analysis: *const Analysis, point: u16) bool {
            if (position.board[point] != EMPTY or position.ko == point) return false;
            const color = position.next_color;
            for (adjacent(point)) |next| {
                const stone = position.board[next];
                if (stone == EMPTY) return true;
                const liberties = analysis.liberties[analysis.key[next]];
                if (stone == color and liberties > 1) return true;
                if (stone == -color and liberties == 1) return true;
            }
            return false;
        }

        /// Writes Moka's getLegalMoves, pass last, and returns how many there are.
        pub fn legalMoves(position: *const Position, analysis: *const Analysis, out: *[MOVES]u16) u16 {
            var count: u16 = 0;
            for (0..AREA) |point| {
                if (isLegal(position, analysis, @intCast(point))) {
                    out[count] = @intCast(point);
                    count += 1;
                }
            }
            out[count] = PASS;
            return count + 1;
        }

        /// The moves the search considers: no pass before MIN_PASS_MOVE_COUNT unless nothing else is legal.
        pub fn candidateMoves(position: *const Position, analysis: *const Analysis, out: *[MOVES]u16) u16 {
            const count = legalMoves(position, analysis, out);
            if (position.move_count >= MIN_PASS_MOVE_COUNT or count == 1) return count;
            return count - 1;
        }

        fn setFeature(features: *[FEATURES]f32, point: usize, plane: Plane, value: f32) void {
            features[point * INPUT_PLANES + @backingInt(plane)] = value;
        }

        /// Moka's encodeStudentFeatures, written as [point][plane].
        pub fn encode(position: *const Position, analysis: *const Analysis, features: *[FEATURES]f32) void {
            @memset(features, 0);
            for (0..AREA) |point| {
                const stone = position.board[point];
                if (stone == EMPTY) continue;
                const is_current = stone == position.next_color;
                setFeature(features, point, if (is_current) .current else .opponent, 1);
                const liberty_plane: ?Plane = switch (analysis.liberties[analysis.key[point]]) {
                    1 => if (is_current) .current_one_liberty else .opponent_one_liberty,
                    2 => if (is_current) .current_two_liberties else .opponent_two_liberties,
                    else => null,
                };
                if (liberty_plane) |plane| setFeature(features, point, plane, 1);
            }
            if (position.ko >= 0) setFeature(features, @intCast(position.ko), .ko, 1);
            encodeRecentMove(features, position.last_move, .last_move, .last_pass);
            encodeRecentMove(features, position.previous_move, .previous_move, .previous_pass);
            const komi: f32 = @floatCast(-@as(f64, KOMI) * @as(f64, @floatFromInt(position.next_color)) / KOMI_NORMALIZATION);
            for (0..AREA) |point| setFeature(features, point, .komi, komi);
        }

        fn encodeRecentMove(features: *[FEATURES]f32, move: i16, move_plane: Plane, pass_plane: Plane) void {
            if (move < 0) return;
            if (move < AREA) {
                setFeature(features, @intCast(move), move_plane, 1);
            } else if (move == PASS) {
                for (0..AREA) |point| setFeature(features, point, pass_plane, 1);
            }
        }

        pub fn isGameOver(position: *const Position) bool {
            return position.pass_count >= GAME_OVER_PASS_COUNT or position.move_count >= MAX_MOVE_COUNT;
        }

        /// Moka's getAreaScore: stones plus empty regions bordered by one color only, minus komi, from Black's side.
        pub fn areaScore(board: *const [AREA]i8) i32 {
            var score: i32 = -KOMI;
            for (board) |stone| score += stone;
            var seen: Points = .empty;
            var pending: [AREA]u16 = undefined;
            for (0..AREA) |start| {
                if (board[start] != EMPTY or seen.isSet(start)) continue;
                var size: i32 = 0;
                var borders_black = false;
                var borders_white = false;
                var pending_len: usize = 1;
                pending[0] = @intCast(start);
                seen.set(start);
                while (pending_len > 0) {
                    pending_len -= 1;
                    size += 1;
                    for (adjacent(pending[pending_len])) |next| {
                        switch (board[next]) {
                            EMPTY => if (!seen.isSet(next)) {
                                seen.set(next);
                                pending[pending_len] = next;
                                pending_len += 1;
                            },
                            BLACK => borders_black = true,
                            else => borders_white = true,
                        }
                    }
                }
                if (borders_black and !borders_white) score += size;
                if (borders_white and !borders_black) score -= size;
            }
            return score;
        }

        const Region = struct {
            points: Points,
            bordering: Points,
            vital: Points,
            contains_opponent: bool,
            internal_points: u16,
        };

        var regions: [AREA]Region = undefined;

        fn markPassAliveArea(board: *const [AREA]i8, color: i8, area: *[AREA]i8) void {
            var group_key: [AREA]u16 = undefined;
            var groups: Points = .empty;
            var seen: Points = .empty;
            for (0..AREA) |point| {
                if (board[point] != color or seen.isSet(point)) continue;
                const group = findGroup(board, @intCast(point));
                seen.setUnion(group.stones);
                groups.set(point);
                var stones = group.stones.iterator(.{});
                while (stones.next()) |stone| group_key[stone] = @intCast(point);
            }
            const has_groups = groups.count() > 0;

            var region_count: usize = 0;
            seen = .empty;
            var pending: [AREA]u16 = undefined;
            for (0..AREA) |start| {
                if (board[start] == color or seen.isSet(start)) continue;
                const region = &regions[region_count];
                region_count += 1;
                region.* = .{ .points = .empty, .bordering = .empty, .vital = .empty, .contains_opponent = false, .internal_points = 0 };
                var has_vital = false;
                var pending_len: usize = 1;
                pending[0] = @intCast(start);
                seen.set(start);
                while (pending_len > 0) {
                    pending_len -= 1;
                    const point = pending[pending_len];
                    region.points.set(point);
                    if (board[point] == -color) region.contains_opponent = true;
                    var adjacent_groups: Points = .empty;
                    for (adjacent(point)) |next| {
                        if (board[next] == color) {
                            adjacent_groups.set(group_key[next]);
                        } else if (!seen.isSet(next)) {
                            seen.set(next);
                            pending[pending_len] = next;
                            pending_len += 1;
                        }
                    }
                    region.bordering.setUnion(adjacent_groups);
                    if (adjacent_groups.count() == 0) region.internal_points += 1;
                    if (board[point] == EMPTY) {
                        if (has_vital) {
                            region.vital.setIntersection(adjacent_groups);
                        } else {
                            region.vital = adjacent_groups;
                            has_vital = true;
                        }
                    }
                }
            }

            var alive_groups = groups;
            var alive_regions: Points = .empty;
            for (0..region_count) |index| alive_regions.set(index);
            while (true) {
                var removed: Points = .empty;
                var keys = alive_groups.iterator(.{});
                while (keys.next()) |key| {
                    var vital_regions: u32 = 0;
                    var indexes = alive_regions.iterator(.{});
                    while (indexes.next()) |index| {
                        if (regions[index].vital.isSet(key)) vital_regions += 1;
                    }
                    if (vital_regions < PASS_ALIVE_REQUIRED_VITAL_REGIONS) removed.set(key);
                }
                if (removed.count() == 0) break;
                alive_groups = alive_groups.differenceWith(removed);
                var indexes = alive_regions.iterator(.{});
                while (indexes.next()) |index| {
                    if (regions[index].bordering.intersectWith(removed).count() > 0) alive_regions.unset(index);
                }
            }

            for (0..AREA) |point| {
                if (board[point] == color and alive_groups.isSet(group_key[point])) area[point] = color;
            }
            for (regions[0..region_count]) |*region| {
                const borders_dead_group = region.bordering.differenceWith(alive_groups).count() > 0;
                const is_pass_alive_region = region.internal_points <= PASS_ALIVE_MAXIMUM_INTERNAL_POINTS and
                    !borders_dead_group and has_groups;
                const is_safe_territory = !region.contains_opponent and !borders_dead_group and has_groups;
                var points = region.points.iterator(.{});
                if (is_pass_alive_region or is_safe_territory) {
                    while (points.next()) |point| area[point] = color;
                } else if (!region.contains_opponent and has_groups) {
                    while (points.next()) |point| {
                        if (area[point] == EMPTY) area[point] = color;
                    }
                }
            }
        }

        /// Moka's getPassAliveArea: Black's marks first, then White's, which overwrite them.
        pub fn passAliveArea(board: *const [AREA]i8) [AREA]i8 {
            var area: [AREA]i8 = @splat(EMPTY);
            markPassAliveArea(board, BLACK, &area);
            markPassAliveArea(board, WHITE, &area);
            return area;
        }

        /// Moka's getBestImmediateCapture: the first move that captures the most stones, from the liberty counts.
        fn bestImmediateCapture(position: *const Position) ?u16 {
            const analysis = analyze(&position.board);
            var best: ?u16 = null;
            var best_count: u32 = 0;
            for (0..AREA) |point| {
                if (position.board[point] != EMPTY or position.ko == point) continue;
                var keys: [4]u16 = undefined;
                var key_count: usize = 0;
                var count: u32 = 0;
                for (adjacent(point)) |next| {
                    if (position.board[next] != -position.next_color) continue;
                    const key = analysis.key[next];
                    if (analysis.liberties[key] != 1 or std.mem.indexOfScalar(u16, keys[0..key_count], key) != null) continue;
                    keys[key_count] = key;
                    key_count += 1;
                    count += analysis.stones[key];
                }
                if (count > best_count) {
                    best_count = count;
                    best = @intCast(point);
                }
            }
            return best;
        }

        /// Moka's getAutomaticallyDeadMoves as a set: pass-dead stones, then what the capture aftermath removes.
        pub fn deadStones(position: *const Position) Points {
            const area = passAliveArea(&position.board);
            var dead: Points = .empty;
            var aftermath = position.*;
            for (0..AREA) |point| {
                const stone = position.board[point];
                if (stone != EMPTY and area[point] == -stone) {
                    dead.set(point);
                    aftermath.board[point] = EMPTY;
                }
            }
            aftermath.pass_count = 0;
            var steps: u32 = 0;
            while (steps < AREA and aftermath.pass_count < GAME_OVER_PASS_COUNT) : (steps += 1) {
                const move = bestImmediateCapture(&aftermath) orelse PASS;
                std.debug.assert(play(&aftermath, move));
            }
            for (0..AREA) |point| {
                if (position.board[point] != EMPTY and aftermath.board[point] == EMPTY) dead.set(point);
            }
            return dead;
        }

        /// Area score after removing the automatically dead stones, from Black's side.
        pub fn finalScore(position: *const Position) i32 {
            var board = position.board;
            const dead = deadStones(position);
            var points = dead.iterator(.{});
            while (points.next()) |point| board[point] = EMPTY;
            return areaScore(&board);
        }
    };
}

const R9 = Rules(9);

fn playAll(position: *R9.Position, moves: []const u16) !void {
    for (moves) |move| try std.testing.expect(R9.play(position, move));
}

test "a lone stone in the corner is captured and ko is set" {
    var position: R9.Position = .{};
    try playAll(&position, &.{ 1, 0, 9 });
    try std.testing.expectEqual(EMPTY, position.board[0]);
    try std.testing.expectEqual(NO_MOVE, position.ko);
    try std.testing.expectEqual(WHITE, position.next_color);
}

test "a single-stone recapture is ko until another move is played" {
    var position: R9.Position = .{};
    try playAll(&position, &.{ 1, 2, 9, 12, 19, 20, 11, 10 });
    try std.testing.expectEqual(@as(i16, 11), position.ko);
    try std.testing.expectEqual(EMPTY, position.board[11]);
    var copy = position;
    try std.testing.expect(!R9.play(&copy, 11));
    try playAll(&position, &.{ 80, 79, 11 });
    try std.testing.expectEqual(@as(i16, 10), position.ko);
    try std.testing.expectEqual(EMPTY, position.board[10]);
    try std.testing.expect(R9.play(&position, R9.PASS));
    try std.testing.expectEqual(NO_MOVE, position.ko);
}

test "suicide is illegal and leaves the position unchanged" {
    var position: R9.Position = .{};
    try playAll(&position, &.{ 1, 40, 9, 41, 80 });
    const before = position;
    try std.testing.expect(!R9.play(&position, 0));
    try std.testing.expectEqual(before, position);
}

test "isLegal agrees with play on every point" {
    var position: R9.Position = .{};
    try playAll(&position, &.{ 1, 0, 9, 40, 10, 41, 18, 42, 2, 50, 11, 51 });
    const analysis = R9.analyze(&position.board);
    for (0..R9.AREA) |point| {
        var copy = position;
        try std.testing.expectEqual(R9.play(&copy, @intCast(point)), R9.isLegal(&position, &analysis, @intCast(point)));
    }
}

test "the empty board scores minus komi" {
    const board: [R9.AREA]i8 = @splat(EMPTY);
    try std.testing.expectEqual(-KOMI, R9.areaScore(&board));
}
