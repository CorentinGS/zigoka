const std = @import("std");
const model_layout = @import("model_layout");
const memory_config = @import("memory_config");
const Geometry = @import("geometry").Geometry;

const config = model_layout.CONFIG;

pub const MAX_BATCH: u32 = 8;
pub const Board = Geometry(config.board_size);
pub const ROWS: u32 = Board.AREA + 1;

pub const INPUT_STRIDE: u32 = ROWS * config.input_planes;
pub const POOL_FEATURES: u32 = 2 * config.bottleneck_channels;
pub const POLICY_FEATURES: u32 = Board.AREA * config.policy_channels;
pub const VALUE_FEATURES: u32 = Board.AREA * config.value_channels;

pub const REORDER_TMP_LEN: u32 = largestConvolution();

fn largestConvolution() u32 {
    var largest: u32 = 0;
    for (model_layout.TENSORS) |tensor| {
        switch (tensor.layout) {
            .conv3x3, .conv1x1 => largest = @max(largest, tensor.len),
            .linear, .bias => {},
        }
    }
    return largest;
}

pub var weights_bin: [model_layout.WEIGHTS_BYTES]u8 align(16) = @splat(0);
pub var weights_f32: [model_layout.WEIGHTS_F32_LEN]f32 align(16) = @splat(0);
pub var reorder_tmp: [REORDER_TMP_LEN]f32 align(16) = @splat(0);

pub var input: [MAX_BATCH * INPUT_STRIDE]f32 align(16) = @splat(0);
pub var trunk: [ROWS * config.trunk_channels]f32 align(16) = @splat(0);
pub var bottleneck_a: [ROWS * config.bottleneck_channels]f32 align(16) = @splat(0);
pub var bottleneck_b: [ROWS * config.bottleneck_channels]f32 align(16) = @splat(0);

pub var pool_features: [POOL_FEATURES]f32 align(16) = @splat(0);
pub var pool_hidden: [config.global_hidden_channels]f32 align(16) = @splat(0);
pub var pool_bias: [config.bottleneck_channels]f32 align(16) = @splat(0);

pub var policy_features: [POLICY_FEATURES]f32 align(16) = @splat(0);
pub var value_features: [VALUE_FEATURES]f32 align(16) = @splat(0);
pub var value_hidden: [config.value_hidden_channels]f32 align(16) = @splat(0);

pub var policy_out: [MAX_BATCH * config.policy_moves]f32 align(16) = @splat(0);
pub var value_out: [MAX_BATCH]f64 align(16) = @splat(0);

pub const STATIC_BYTES: u64 = @sizeOf(@TypeOf(weights_bin)) + @sizeOf(@TypeOf(weights_f32)) +
    @sizeOf(@TypeOf(reorder_tmp)) + @sizeOf(@TypeOf(input)) + @sizeOf(@TypeOf(trunk)) +
    @sizeOf(@TypeOf(bottleneck_a)) + @sizeOf(@TypeOf(bottleneck_b)) +
    @sizeOf(@TypeOf(pool_features)) + @sizeOf(@TypeOf(pool_hidden)) + @sizeOf(@TypeOf(pool_bias)) +
    @sizeOf(@TypeOf(policy_features)) + @sizeOf(@TypeOf(value_features)) +
    @sizeOf(@TypeOf(value_hidden)) + @sizeOf(@TypeOf(policy_out)) + @sizeOf(@TypeOf(value_out));

pub const MEMORY_BYTES: u64 = @as(u64, memory_config.memory_pages) * 64 * 1024;
pub const USED_BYTES: u64 = STATIC_BYTES + memory_config.stack_bytes;

comptime {
    if (USED_BYTES * 5 > MEMORY_BYTES * 4) {
        @compileError(std.fmt.comptimePrint(
            "{d} pages ({d} bytes) leave less than 25% headroom over {d} bytes of buffers and stack",
            .{ memory_config.memory_pages, MEMORY_BYTES, USED_BYTES },
        ));
    }
}

test "buffer sizes follow the memory table" {
    try std.testing.expectEqual(@as(usize, 113_648), weights_bin.len);
    try std.testing.expectEqual(@as(usize, 421_412), @sizeOf(@TypeOf(weights_f32)));
    try std.testing.expectEqual(@as(usize, 13_824), @sizeOf(@TypeOf(reorder_tmp)));
    try std.testing.expectEqual(@as(usize, 31_488), @sizeOf(@TypeOf(input)));
    try std.testing.expectEqual(@as(usize, 10_496), @sizeOf(@TypeOf(trunk)));
    try std.testing.expectEqual(@as(usize, 5_248), @sizeOf(@TypeOf(bottleneck_a)));
    try std.testing.expectEqual(@as(usize, 128 + 32 + 64), @sizeOf(@TypeOf(pool_features)) +
        @sizeOf(@TypeOf(pool_hidden)) + @sizeOf(@TypeOf(pool_bias)));
    try std.testing.expectEqual(@as(usize, 1_296 + 648 + 128), @sizeOf(@TypeOf(policy_features)) +
        @sizeOf(@TypeOf(value_features)) + @sizeOf(@TypeOf(value_hidden)));
    try std.testing.expectEqual(@as(usize, 2_624), @sizeOf(@TypeOf(policy_out)));
    try std.testing.expectEqual(@as(usize, 64), @sizeOf(@TypeOf(value_out)));
}

test "the configured page count is the smallest one with 25% headroom" {
    const page: u64 = 64 * 1024;
    const smaller = MEMORY_BYTES - page;
    try std.testing.expect(USED_BYTES * 5 > smaller * 4);
}
