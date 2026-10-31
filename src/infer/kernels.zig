const std = @import("std");

pub const Lane = @Vector(4, f32);

pub inline fn relu(x: f32) f32 {
    return if (x <= 0) 0 else x;
}

pub inline fn reluLane(x: Lane) Lane {
    const zero: Lane = @splat(0);
    return @select(f32, x <= zero, zero, x);
}

pub const Store = enum {
    relu,
    add_to_trunk_relu,
};

fn Accumulator(comptime outputs: usize) type {
    return if (outputs % 4 == 0) [outputs / 4]Lane else [outputs]f32;
}

inline fn accumulate(
    comptime taps: usize,
    comptime inputs: usize,
    comptime outputs: usize,
    rows: *const [taps]u16,
    input: []const f32,
    weights: []const f32,
    bias: []const f32,
) Accumulator(outputs) {
    var acc: Accumulator(outputs) = undefined;
    if (comptime outputs % 4 == 0) {
        inline for (0..outputs / 4) |lane| acc[lane] = bias[4 * lane ..][0..4].*;
        for (rows, 0..) |row, tap| {
            const x_row = input[@as(usize, row) * inputs ..][0..inputs];
            const w_tap = weights[tap * inputs * outputs ..][0 .. inputs * outputs];
            for (x_row, 0..) |x, channel| {
                const xs: Lane = @splat(x);
                const w = w_tap[channel * outputs ..][0..outputs];
                inline for (0..outputs / 4) |lane| {
                    const w_lane: Lane = w[4 * lane ..][0..4].*;
                    acc[lane] = acc[lane] + xs * w_lane;
                }
            }
        }
    } else {
        acc = bias[0..outputs].*;
        for (rows, 0..) |row, tap| {
            const x_row = input[@as(usize, row) * inputs ..][0..inputs];
            const w_tap = weights[tap * inputs * outputs ..][0 .. inputs * outputs];
            for (x_row, 0..) |x, channel| {
                const w = w_tap[channel * outputs ..][0..outputs];
                inline for (0..outputs) |output| acc[output] = acc[output] + x * w[output];
            }
        }
    }
    return acc;
}

/// Positions and output lanes computed together by `accumulateTile`: each weight vector serves every
/// position of the tile, and the accumulators of a tile stay in registers.
const Tile = struct { positions: usize, lanes: usize };

fn tileFor(comptime outputs: usize) Tile {
    if (outputs % 4 != 0) return .{ .positions = 1, .lanes = 0 };
    for (TILES) |entry| {
        if (entry.outputs == outputs) return entry.tile;
    }
    return .{ .positions = 1, .lanes = outputs / 4 };
}

/// Measured in V8: past about 9 accumulators, the JIT spills them and a tile gets slower.
const TILES = [_]struct { outputs: usize, tile: Tile }{
    .{ .outputs = 4, .tile = .{ .positions = 2, .lanes = 1 } },
    .{ .outputs = 8, .tile = .{ .positions = 2, .lanes = 2 } },
    .{ .outputs = 12, .tile = .{ .positions = 2, .lanes = 3 } },
    .{ .outputs = 16, .tile = .{ .positions = 2, .lanes = 4 } },
    .{ .outputs = 24, .tile = .{ .positions = 3, .lanes = 3 } },
    .{ .outputs = 32, .tile = .{ .positions = 2, .lanes = 4 } },
    .{ .outputs = 48, .tile = .{ .positions = 3, .lanes = 3 } },
};

inline fn accumulateTile(
    comptime positions: usize,
    comptime taps: usize,
    comptime inputs: usize,
    comptime outputs: usize,
    comptime lanes: usize,
    comptime first_lane: usize,
    rows: [positions]*const [taps]u16,
    input: []const f32,
    weights: []const f32,
    bias: []const f32,
) [positions][lanes]Lane {
    var acc: [positions][lanes]Lane = undefined;
    inline for (0..positions) |p| {
        inline for (0..lanes) |lane| acc[p][lane] = bias[4 * (first_lane + lane) ..][0..4].*;
    }
    for (0..taps) |tap| {
        var x_rows: [positions]*const [inputs]f32 = undefined;
        inline for (0..positions) |p| x_rows[p] = input[@as(usize, rows[p][tap]) * inputs ..][0..inputs];
        const w_tap = weights[tap * inputs * outputs ..][0 .. inputs * outputs];
        for (0..inputs) |channel| {
            var xs: [positions]Lane = undefined;
            inline for (0..positions) |p| xs[p] = @splat(x_rows[p][channel]);
            const w = w_tap[channel * outputs + 4 * first_lane ..][0 .. 4 * lanes];
            inline for (0..lanes) |lane| {
                const w_lane: Lane = w[4 * lane ..][0..4].*;
                inline for (0..positions) |p| acc[p][lane] = acc[p][lane] + xs[p] * w_lane;
            }
        }
    }
    return acc;
}

inline fn storeLanes(comptime lanes: usize, comptime store: Store, acc: [lanes]Lane, out: *[4 * lanes]f32) void {
    inline for (0..lanes) |lane| {
        const target = out[4 * lane ..][0..4];
        target.* = switch (store) {
            .relu => reluLane(acc[lane]),
            .add_to_trunk_relu => reluLane(@as(Lane, target.*) + acc[lane]),
        };
    }
}

/// Convolution over every intersection of `Board`, vectorized over output channels.
/// `input` holds `Board.AREA + 1` rows when `taps == 9`; the last row is the zero sentinel.
/// `weights` use the `[tap][in][out]` layout produced by `weights.reorder`.
pub fn convolve(
    comptime Board: type,
    comptime taps: usize,
    comptime inputs: usize,
    comptime outputs: usize,
    comptime store: Store,
    input: []const f32,
    weights: []const f32,
    bias: []const f32,
    output: []f32,
) void {
    comptime std.debug.assert(taps == 9 or taps == 1);
    std.debug.assert(weights.len == taps * inputs * outputs);
    std.debug.assert(output.len >= Board.AREA * outputs);
    const tile = comptime tileFor(outputs);
    const tiled = comptime tile.lanes > 0 and (tile.positions > 1 or tile.lanes < outputs / 4);
    const blocked = if (tiled) Board.AREA - Board.AREA % tile.positions else 0;
    var first: usize = 0;
    if (comptime tiled) {
        comptime std.debug.assert((outputs / 4) % tile.lanes == 0);
        while (first < blocked) : (first += tile.positions) {
            var rows: [tile.positions]*const [taps]u16 = undefined;
            inline for (0..tile.positions) |p| rows[p] = if (taps == 9) &Board.CONV_NEIGHBORS[first + p] else &.{@intCast(first + p)};
            inline for (0..outputs / 4 / tile.lanes) |chunk| {
                const first_lane = chunk * tile.lanes;
                const acc = accumulateTile(tile.positions, taps, inputs, outputs, tile.lanes, first_lane, rows, input, weights, bias);
                inline for (0..tile.positions) |p| {
                    storeLanes(tile.lanes, store, acc[p], output[(first + p) * outputs + 4 * first_lane ..][0 .. 4 * tile.lanes]);
                }
            }
        }
    }
    for (first..Board.AREA) |position| {
        const rows: *const [taps]u16 = if (taps == 9) &Board.CONV_NEIGHBORS[position] else &.{@intCast(position)};
        const acc = accumulate(taps, inputs, outputs, rows, input, weights, bias);
        const out = output[position * outputs ..][0..outputs];
        if (comptime outputs % 4 == 0) {
            storeLanes(outputs / 4, store, acc, out);
        } else {
            inline for (0..outputs) |index| {
                out[index] = switch (store) {
                    .relu => relu(acc[index]),
                    .add_to_trunk_relu => relu(out[index] + acc[index]),
                };
            }
        }
    }
}

const testing = std.testing;
const Geometry = @import("geometry").Geometry;

fn scalarReference(
    comptime Board: type,
    comptime taps: usize,
    comptime inputs: usize,
    comptime outputs: usize,
    comptime store: Store,
    input: []const f32,
    weights: []const f32,
    bias: []const f32,
    output: []f32,
) void {
    for (0..Board.AREA) |position| {
        for (0..outputs) |o| {
            var acc = bias[o];
            for (0..taps) |tap| {
                const row: usize = if (taps == 9) Board.CONV_NEIGHBORS[position][tap] else position;
                for (0..inputs) |c| acc = acc + input[row * inputs + c] * weights[(tap * inputs + c) * outputs + o];
            }
            const index = position * outputs + o;
            output[index] = switch (store) {
                .relu => relu(acc),
                .add_to_trunk_relu => relu(output[index] + acc),
            };
        }
    }
}

fn expectSameAsScalar(comptime board_size: u16, comptime taps: usize, comptime inputs: usize, comptime outputs: usize, comptime store: Store) !void {
    const Board = Geometry(board_size);
    var prng: std.Random.DefaultPrng = .init(taps * 1000 + inputs * 10 + outputs);
    const random = prng.random();
    var input: [(Board.AREA + 1) * inputs]f32 = undefined;
    for (&input) |*x| x.* = random.float(f32) * 2 - 1;
    @memset(input[Board.AREA * inputs ..], 0);
    var weights: [taps * inputs * outputs]f32 = undefined;
    for (&weights) |*w| w.* = random.float(f32) - 0.5;
    var bias: [outputs]f32 = undefined;
    for (&bias) |*b| b.* = random.float(f32) - 0.5;
    var start: [Board.AREA * outputs]f32 = undefined;
    for (&start) |*x| x.* = random.float(f32) - 0.5;

    var vector = start;
    var scalar = start;
    convolve(Board, taps, inputs, outputs, store, &input, &weights, &bias, &vector);
    scalarReference(Board, taps, inputs, outputs, store, &input, &weights, &bias, &scalar);
    try testing.expectEqualSlices(u32, @ptrCast(&scalar), @ptrCast(&vector));
}

test "SIMD convolutions match the scalar order bit for bit" {
    inline for ([_]u16{ 9, 19 }) |board_size| {
        try expectSameAsScalar(board_size, 9, 12, 32, .relu);
        try expectSameAsScalar(board_size, 9, 16, 16, .relu);
        try expectSameAsScalar(board_size, 1, 32, 16, .relu);
        try expectSameAsScalar(board_size, 1, 16, 32, .add_to_trunk_relu);
        try expectSameAsScalar(board_size, 1, 32, 4, .relu);
        try expectSameAsScalar(board_size, 1, 32, 2, .relu);
        try expectSameAsScalar(board_size, 9, 12, 48, .relu);
        try expectSameAsScalar(board_size, 9, 24, 24, .relu);
        try expectSameAsScalar(board_size, 1, 48, 24, .relu);
        try expectSameAsScalar(board_size, 1, 24, 48, .add_to_trunk_relu);
        try expectSameAsScalar(board_size, 1, 48, 4, .relu);
    }
}

test "relu follows Math.max(0, x)" {
    try testing.expectEqual(@as(u32, 0), @as(u32, @bitCast(relu(-0.0))));
    try testing.expectEqual(@as(u32, 0), @as(u32, @bitCast(relu(-1.5))));
    try testing.expectEqual(@as(f32, 2.5), relu(2.5));
    try testing.expect(std.math.isNan(relu(std.math.nan(f32))));
    const lanes = reluLane(.{ -0.0, -3, std.math.nan(f32), 4 });
    try testing.expectEqual(@as(u32, 0), @as(u32, @bitCast(lanes[0])));
    try testing.expectEqual(@as(f32, 0), lanes[1]);
    try testing.expect(std.math.isNan(lanes[2]));
    try testing.expectEqual(@as(f32, 4), lanes[3]);
}
