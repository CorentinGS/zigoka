const std = @import("std");
const relu = @import("kernels.zig").relu;

/// Mean then maximum of each channel over `area` intersections, rounded like Moka's
/// `getSpatialMeanAndMaximumValues`: the mean is summed in f64 and stored in f32 at each step.
pub fn poolMeanMax(comptime area: usize, comptime channels: usize, values: []const f32, out: []f32) void {
    std.debug.assert(out.len == 2 * channels);
    const count: f64 = @floatFromInt(area);
    @memset(out[0..channels], 0);
    @memset(out[channels..], -std.math.inf(f32));
    for (0..area) |position| {
        for (0..channels) |channel| {
            const value = values[position * channels + channel];
            out[channel] = @floatCast(@as(f64, out[channel]) + @as(f64, value) / count);
            const maximum = &out[channels + channel];
            if (value > maximum.* or value != value) maximum.* = value;
        }
    }
}

const LINEAR_ROWS = 4;
const RowSums = @Vector(LINEAR_ROWS, f64);

/// `out = bias + weights · input`, accumulated in f64 and stored in f32, like Moka's `applyLinear`.
/// `weights` keep the `[out][in]` layout of the manifest. Rows are computed `LINEAR_ROWS` at a
/// time so their sums run in parallel; each row keeps Moka's order of additions.
pub fn linear(comptime inputs: usize, comptime outputs: usize, input: []const f32, weights: []const f32, bias: []const f32, out: []f32) void {
    std.debug.assert(input.len == inputs and weights.len == inputs * outputs);
    const blocked = outputs - outputs % LINEAR_ROWS;
    var first: usize = 0;
    while (first < blocked) : (first += LINEAR_ROWS) {
        var sums: RowSums = undefined;
        inline for (0..LINEAR_ROWS) |row| sums[row] = bias[first + row];
        for (input[0..inputs], 0..) |x, column| {
            var row_weights: RowSums = undefined;
            inline for (0..LINEAR_ROWS) |row| row_weights[row] = weights[(first + row) * inputs + column];
            sums += @as(RowSums, @splat(@as(f64, x))) * row_weights;
        }
        inline for (0..LINEAR_ROWS) |row| out[first + row] = @floatCast(sums[row]);
    }
    for (blocked..outputs) |output| {
        const row = weights[output * inputs ..][0..inputs];
        var sum: f64 = bias[output];
        for (input[0..inputs], row) |x, w| sum += @as(f64, x) * @as(f64, w);
        out[output] = @floatCast(sum);
    }
}

pub fn reluInPlace(values: []f32) void {
    for (values) |*value| value.* = relu(value.*);
}

pub fn tanh(x: f32) f64 {
    return std.math.tanh(@as(f64, x));
}

const testing = std.testing;

test "the pooled maximum propagates NaN like Math.max" {
    var values = [_]f32{ 1, std.math.nan(f32), 3, 2 };
    var out: [4]f32 = undefined;
    poolMeanMax(2, 2, &values, &out);
    try testing.expectEqual(@as(f32, 2), out[0]);
    try testing.expect(std.math.isNan(out[1]));
    try testing.expectEqual(@as(f32, 3), out[2]);
    try testing.expect(std.math.isNan(out[3]));
}

test "linear accumulates in f64 before rounding to f32" {
    const input = [_]f32{ 1.0, 1.0e-8, -1.0 };
    const weights = [_]f32{ 1.0, 1.0, 1.0 };
    const bias = [_]f32{0};
    var out: [1]f32 = undefined;
    linear(3, 1, &input, &weights, &bias, &out);
    try testing.expectEqual(@as(f32, @floatCast(@as(f64, @as(f32, 1.0e-8)))), out[0]);
}

test "linear gives the same bits as one sequential sum per row" {
    const inputs = 13;
    const outputs = 2 * LINEAR_ROWS + 3;
    var prng = std.Random.DefaultPrng.init(19);
    const random = prng.random();
    var input: [inputs]f32 = undefined;
    var weights: [inputs * outputs]f32 = undefined;
    var bias: [outputs]f32 = undefined;
    for (&input) |*x| x.* = random.float(f32) * 2 - 1;
    for (&weights) |*w| w.* = (random.float(f32) * 2 - 1) * 1.0e3;
    for (&bias) |*b| b.* = random.float(f32);
    var out: [outputs]f32 = undefined;
    linear(inputs, outputs, &input, &weights, &bias, &out);
    for (0..outputs) |output| {
        var sum: f64 = bias[output];
        for (0..inputs) |column| sum += @as(f64, input[column]) * @as(f64, weights[output * inputs + column]);
        try testing.expectEqual(@as(u32, @bitCast(@as(f32, @floatCast(sum)))), @as(u32, @bitCast(out[output])));
    }
}
