const std = @import("std");
const buffers = @import("buffers.zig");
const weights = @import("weights.zig");
const network = @import("network.zig");
const Status = @import("exports.zig").Status;

pub const panic = std.debug.no_panic;

comptime {
    _ = @import("exports.zig");
}

export fn forward_trunk(count: u32) u32 {
    if (count < 1 or count > buffers.MAX_BATCH) return @backingInt(Status.bad_batch_count);
    if (!weights.loaded) return @backingInt(Status.weights_not_loaded);
    for (0..count) |k| network.forwardTrunk(k);
    return @backingInt(Status.ok);
}
