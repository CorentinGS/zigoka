const std = @import("std");
const model_layout = @import("model_layout");
const buffers = @import("buffers.zig");
const kernels = @import("kernels.zig");
const dense = @import("dense.zig");

const config = model_layout.CONFIG;
const Board = buffers.Board;
const AREA = Board.AREA;
const TRUNK = config.trunk_channels;
const BOTTLENECK = config.bottleneck_channels;

const Span = struct { offset: u32, len: u32 };

fn span(comptime name: []const u8) Span {
    const tensor = model_layout.tensor(name);
    return .{ .offset = tensor.f32_offset, .len = tensor.len };
}

fn view(s: Span) []const f32 {
    return buffers.weights_f32[s.offset..][0..s.len];
}

const Layer = struct { weight: Span, bias: Span };

fn layer(comptime name: []const u8) Layer {
    return .{ .weight = span(name ++ ".weight"), .bias = span(name ++ ".bias") };
}

const Block = struct {
    reduce: Layer,
    first: Layer,
    global: ?struct { hidden: Layer, output: Layer },
    second: Layer,
    expand: Layer,
};

const BLOCKS: [config.residual_blocks]Block = blocks: {
    @setEvalBranchQuota(100_000);
    var table: [config.residual_blocks]Block = undefined;
    for (&table, 0..) |*block, index| {
        const prefix = std.fmt.comptimePrint("residual.{d}.", .{index});
        const is_global = config.global_block_interval > 0 and (index + 1) % config.global_block_interval == 0;
        block.* = .{
            .reduce = layer(prefix ++ "reduce"),
            .first = layer(prefix ++ "first"),
            .global = if (is_global) .{ .hidden = layer(prefix ++ "global.hidden"), .output = layer(prefix ++ "global.output") } else null,
            .second = layer(prefix ++ "second"),
            .expand = layer(prefix ++ "expand"),
        };
    }
    break :blocks table;
};

const STEM = layer("stem");
const POLICY_CONVOLUTION = layer("policy.convolution");
const POLICY_LINEAR = layer("policy.linear");
const VALUE_CONVOLUTION = layer("value.convolution");
const VALUE_HIDDEN = layer("value.hidden");
const VALUE_OUTPUT = layer("value.output");

fn residualBlock(block: Block) void {
    const a = &buffers.bottleneck_a;
    const b = &buffers.bottleneck_b;
    kernels.convolve(Board, 1, TRUNK, BOTTLENECK, .relu, &buffers.trunk, view(block.reduce.weight), view(block.reduce.bias), a);
    kernels.convolve(Board, 9, BOTTLENECK, BOTTLENECK, .relu, a, view(block.first.weight), view(block.first.bias), b);
    if (block.global) |global| {
        dense.poolMeanMax(AREA, BOTTLENECK, b, &buffers.pool_features);
        dense.linear(buffers.POOL_FEATURES, config.global_hidden_channels, &buffers.pool_features, view(global.hidden.weight), view(global.hidden.bias), &buffers.pool_hidden);
        dense.reluInPlace(&buffers.pool_hidden);
        dense.linear(config.global_hidden_channels, BOTTLENECK, &buffers.pool_hidden, view(global.output.weight), view(global.output.bias), &buffers.pool_bias);
        for (0..AREA) |position| {
            for (b[position * BOTTLENECK ..][0..BOTTLENECK], buffers.pool_bias) |*value, bias| value.* = value.* + bias;
        }
    }
    kernels.convolve(Board, 9, BOTTLENECK, BOTTLENECK, .relu, b, view(block.second.weight), view(block.second.bias), a);
    kernels.convolve(Board, 1, BOTTLENECK, TRUNK, .add_to_trunk_relu, a, view(block.expand.weight), view(block.expand.bias), &buffers.trunk);
}

/// Runs the network on position `k` of the batch. Returns false if an output is not finite.
pub fn forward(k: usize) bool {
    const input = buffers.input[k * buffers.INPUT_STRIDE ..][0..buffers.INPUT_STRIDE];
    kernels.convolve(Board, 9, config.input_planes, TRUNK, .relu, input, view(STEM.weight), view(STEM.bias), &buffers.trunk);

    for (BLOCKS) |block| residualBlock(block);

    const logits = buffers.policy_out[k * config.policy_moves ..][0..config.policy_moves];
    kernels.convolve(Board, 1, TRUNK, config.policy_channels, .relu, &buffers.trunk, view(POLICY_CONVOLUTION.weight), view(POLICY_CONVOLUTION.bias), &buffers.policy_features);
    dense.linear(buffers.POLICY_FEATURES, config.policy_moves, &buffers.policy_features, view(POLICY_LINEAR.weight), view(POLICY_LINEAR.bias), logits);

    kernels.convolve(Board, 1, TRUNK, config.value_channels, .relu, &buffers.trunk, view(VALUE_CONVOLUTION.weight), view(VALUE_CONVOLUTION.bias), &buffers.value_features);
    dense.linear(buffers.VALUE_FEATURES, config.value_hidden_channels, &buffers.value_features, view(VALUE_HIDDEN.weight), view(VALUE_HIDDEN.bias), &buffers.value_hidden);
    dense.reluInPlace(&buffers.value_hidden);
    var value_output: [1]f32 = undefined;
    dense.linear(config.value_hidden_channels, 1, &buffers.value_hidden, view(VALUE_OUTPUT.weight), view(VALUE_OUTPUT.bias), &value_output);
    const value = dense.tanh(value_output[0]);
    buffers.value_out[k] = value;

    for (logits) |logit| {
        if (!std.math.isFinite(logit)) return false;
    }
    return std.math.isFinite(value);
}

const testing = std.testing;
const weights = @import("weights.zig");
const build_options = @import("build_options");

fn expectSentinelRowZero(rows: []const f32, channels: usize) !void {
    for (rows[AREA * channels ..][0..channels]) |value| try testing.expectEqual(@as(u32, 0), @as(u32, @bitCast(value)));
}

test "forward keeps every sentinel row at zero" {
    @memcpy(&buffers.weights_bin, @embedFile("go_model_bin"));
    weights.load();

    var path_buffer: [std.fs.max_path_bytes]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buffer, "{s}/validation/features.f32", .{build_options.fixtures_dir});
    const features = try std.Io.Dir.cwd().readFileAlloc(testing.io, path, testing.allocator, .limited(64 * 1024 * 1024));
    defer testing.allocator.free(features);

    const feature_count = AREA * config.input_planes;
    for (0..buffers.MAX_BATCH) |k| {
        const input = buffers.input[k * buffers.INPUT_STRIDE ..][0..feature_count];
        for (input, 0..) |*x, index| x.* = @bitCast(std.mem.readInt(u32, features[4 * (k * feature_count + index) ..][0..4], .little));
        try testing.expect(forward(k));
    }

    for (0..buffers.MAX_BATCH) |k| try expectSentinelRowZero(buffers.input[k * buffers.INPUT_STRIDE ..][0..buffers.INPUT_STRIDE], config.input_planes);
    try expectSentinelRowZero(&buffers.trunk, TRUNK);
    try expectSentinelRowZero(&buffers.bottleneck_a, BOTTLENECK);
    try expectSentinelRowZero(&buffers.bottleneck_b, BOTTLENECK);
}
