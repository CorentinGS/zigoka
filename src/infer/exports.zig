const std = @import("std");
const model_layout = @import("model_layout");

pub const panic = std.debug.no_panic;

const MAX_BATCH: u32 = 8;

export fn max_batch() u32 {
    return MAX_BATCH;
}

export fn weights_capacity() u32 {
    return model_layout.WEIGHTS_BYTES;
}
