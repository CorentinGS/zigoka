const std = @import("std");
const model_layout = @import("model_layout");
const buffers = @import("buffers.zig");
const weights = @import("weights.zig");
const network = @import("network.zig");

pub const panic = std.debug.no_panic;

const Status = enum(u32) {
    ok = 0,
    weights_not_loaded = 1,
    bad_batch_count = 2,
    non_finite_output = 3,
};

const expected_sha256: [32]u8 = model_layout.SHA256;

export fn weights_ptr() [*]u8 {
    return &buffers.weights_bin;
}

export fn weights_capacity() u32 {
    return model_layout.WEIGHTS_BYTES;
}

export fn expected_sha256_ptr() [*]const u8 {
    return &expected_sha256;
}

export fn load_weights() void {
    weights.load();
}

export fn input_ptr() [*]f32 {
    return &buffers.input;
}

export fn policy_ptr() [*]f32 {
    return &buffers.policy_out;
}

export fn value_ptr() [*]f64 {
    return &buffers.value_out;
}

export fn max_batch() u32 {
    return buffers.MAX_BATCH;
}

export fn infer(count: u32) Status {
    if (count < 1 or count > buffers.MAX_BATCH) return .bad_batch_count;
    if (!weights.loaded) return .weights_not_loaded;
    var status: Status = .ok;
    for (0..count) |k| {
        if (!network.forward(k)) status = .non_finite_output;
    }
    return status;
}
