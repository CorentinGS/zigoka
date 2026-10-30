const std = @import("std");
const model_layout = @import("model_layout");

test {
    _ = @import("geometry.zig");
}

test "the generated layout matches the Moka model" {
    try std.testing.expectEqual(@as(u32, 113_648), model_layout.WEIGHTS_BYTES);
    try std.testing.expectEqual(@as(u32, 105_353), model_layout.WEIGHTS_F32_LEN);
    try std.testing.expectEqual(@as(u16, 9), model_layout.CONFIG.board_size);
    const stem = model_layout.tensor("stem.weight");
    try std.testing.expectEqual(model_layout.Layout.conv3x3, stem.layout);
    try std.testing.expectEqual(@as(u32, 32 * 3 * 3 * 12), stem.len);
    const last = model_layout.TENSORS[model_layout.TENSORS.len - 1];
    try std.testing.expectEqual(model_layout.WEIGHTS_F32_LEN, last.f32_offset + last.len);
}
