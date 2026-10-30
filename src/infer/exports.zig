const std = @import("std");
const model_layout = @import("model_layout");
const buffers = @import("buffers.zig");

pub const panic = std.debug.no_panic;

export fn max_batch() u32 {
    return buffers.MAX_BATCH;
}

export fn weights_capacity() u32 {
    return model_layout.WEIGHTS_BYTES;
}
