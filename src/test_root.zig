const std = @import("std");
const model_layout = @import("model_layout");

test {
    _ = @import("infer/buffers.zig");
    _ = @import("infer/weights.zig");
    _ = @import("infer/kernels.zig");
    _ = @import("infer/dense.zig");
    _ = @import("infer/network.zig");
}

const build_options = @import("build_options");

test "the generated layout matches the Moka model" {
    if (!std.mem.eql(u8, build_options.model_id, "moka-9")) return error.SkipZigTest;
    try std.testing.expectEqual(@as(u32, 113_648), model_layout.WEIGHTS_BYTES);
    try std.testing.expectEqual(@as(u32, 105_353), model_layout.WEIGHTS_F32_LEN);
    try std.testing.expectEqual(@as(u16, 9), model_layout.CONFIG.board_size);
    const stem = model_layout.tensor("stem.weight");
    try std.testing.expectEqual(model_layout.Layout.conv3x3, stem.layout);
    try std.testing.expectEqual(@as(u32, 32 * 3 * 3 * 12), stem.len);
}

test "the generated layout is consistent" {
    const last = model_layout.TENSORS[model_layout.TENSORS.len - 1];
    try std.testing.expectEqual(model_layout.WEIGHTS_F32_LEN, last.f32_offset + last.len);
    const board_size: u32 = model_layout.CONFIG.board_size;
    try std.testing.expectEqual(board_size * board_size + 1, model_layout.CONFIG.policy_moves);
}
