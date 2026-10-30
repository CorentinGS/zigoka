const std = @import("std");
const model_layout = @import("model_layout");
const buffers = @import("buffers.zig");

pub var loaded: bool = false;

fn readF32(bytes: []const u8, offset: usize) f32 {
    return @bitCast(std.mem.readInt(u32, bytes[offset..][0..4], .little));
}

fn readQuantized(comptime kind: model_layout.Kind, bin: []const u8, data_offset: usize, index: usize) f32 {
    return switch (kind) {
        .int8 => @floatFromInt(@as(i8, @bitCast(bin[data_offset + index]))),
        .int16 => @floatFromInt(std.mem.readInt(i16, bin[data_offset + 2 * index ..][0..2], .little)),
        .float32 => @compileError("float32 tensors are not quantized"),
    };
}

fn dequantizeQuantized(comptime kind: model_layout.Kind, tensor: model_layout.Tensor, bin: []const u8, values: []f32) void {
    const scale_offset = tensor.scale_offset.?;
    const per_output = tensor.len / tensor.shape[0];
    for (values, 0..) |*value, index| {
        const scale = readF32(bin, scale_offset + 4 * (index / per_output));
        value.* = readQuantized(kind, bin, tensor.data_offset, index) * scale;
    }
}

fn dequantizeTensor(tensor: model_layout.Tensor, bin: []const u8, values: []f32) void {
    switch (tensor.kind) {
        .float32 => for (values, 0..) |*value, index| {
            value.* = readF32(bin, tensor.data_offset + 4 * index);
        },
        .int8 => dequantizeQuantized(.int8, tensor, bin, values),
        .int16 => dequantizeQuantized(.int16, tensor, bin, values),
    }
}

pub fn dequantize(bin: []const u8, out: []f32) void {
    std.debug.assert(bin.len == model_layout.WEIGHTS_BYTES);
    std.debug.assert(out.len == model_layout.WEIGHTS_F32_LEN);
    for (model_layout.TENSORS) |tensor| {
        dequantizeTensor(tensor, bin, out[tensor.f32_offset..][0..tensor.len]);
    }
}

pub fn reorder(weights: []f32, tmp: []f32) void {
    std.debug.assert(weights.len == model_layout.WEIGHTS_F32_LEN);
    for (model_layout.TENSORS) |tensor| {
        switch (tensor.layout) {
            .conv3x3, .conv1x1 => {},
            .linear, .bias => continue,
        }
        const outputs = tensor.shape[0];
        const taps = tensor.shape[1] * tensor.shape[2];
        const inputs = tensor.shape[3];
        const values = weights[tensor.f32_offset..][0..tensor.len];
        const original = tmp[0..tensor.len];
        @memcpy(original, values);
        for (0..outputs) |output| {
            for (0..taps) |tap| {
                for (0..inputs) |input| {
                    values[(tap * inputs + input) * outputs + output] =
                        original[(output * taps + tap) * inputs + input];
                }
            }
        }
    }
}

pub fn load() void {
    dequantize(&buffers.weights_bin, &buffers.weights_f32);
    reorder(&buffers.weights_f32, &buffers.reorder_tmp);
    loaded = true;
}

const testing = std.testing;
const build_options = @import("build_options");
const go_model_bin = @embedFile("go_model_bin");

fn readReferenceTensor(gpa: std.mem.Allocator, name: []const u8) ![]u8 {
    var path_buffer: [std.fs.max_path_bytes]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buffer, "{s}/tensors/{s}.f32", .{ build_options.fixtures_dir, name });
    return std.Io.Dir.cwd().readFileAlloc(testing.io, path, gpa, .limited(1024 * 1024)) catch |err| {
        std.debug.print("cannot read {s} ({t}); run `pnpm oracle` before `zig build test`\n", .{ path, err });
        return err;
    };
}

test "dequantized weights are bit for bit those of Moka" {
    const out = try testing.allocator.alloc(f32, model_layout.WEIGHTS_F32_LEN);
    defer testing.allocator.free(out);
    dequantize(go_model_bin, out);

    for (model_layout.TENSORS) |tensor| {
        const reference = try readReferenceTensor(testing.allocator, tensor.name);
        defer testing.allocator.free(reference);
        if (reference.len != 4 * tensor.len) {
            std.debug.print("{s}: {d} reference bytes, {d} expected\n", .{ tensor.name, reference.len, 4 * tensor.len });
            return error.TestUnexpectedResult;
        }
        for (out[tensor.f32_offset..][0..tensor.len], 0..) |value, index| {
            const expected = std.mem.readInt(u32, reference[4 * index ..][0..4], .little);
            const actual: u32 = @bitCast(value);
            if (actual != expected) {
                std.debug.print("{s}[{d}]: 0x{x:0>8}, Moka has 0x{x:0>8}\n", .{ tensor.name, index, actual, expected });
                return error.TestUnexpectedResult;
            }
        }
    }
}

test "int16 values are read little-endian, negative ones included" {
    const bin = [_]u8{
        0x18, 0xfc, 0x01, 0x00,
        0xff, 0x7f, 0x00, 0x80,
        0x00, 0x00, 0x00, 0x3f,
        0x00, 0x00, 0x80, 0x40,
    };
    const tensor: model_layout.Tensor = .{
        .name = "synthetic",
        .kind = .int16,
        .layout = .linear,
        .shape = &.{ 2, 2 },
        .data_offset = 0,
        .scale_offset = 8,
        .f32_offset = 0,
        .len = 4,
    };
    var values: [4]f32 = undefined;
    dequantizeTensor(tensor, &bin, &values);
    try testing.expectEqualSlices(f32, &.{ -500, 0.5, 131_068, -131_072 }, &values);
}

test "reorder moves convolution weights to [tap][in][out] and leaves the rest alone" {
    const original = try testing.allocator.alloc(f32, model_layout.WEIGHTS_F32_LEN);
    defer testing.allocator.free(original);
    const reordered = try testing.allocator.alloc(f32, model_layout.WEIGHTS_F32_LEN);
    defer testing.allocator.free(reordered);
    var tmp: [buffers.REORDER_TMP_LEN]f32 = undefined;

    dequantize(go_model_bin, original);
    @memcpy(reordered, original);
    reorder(reordered, &tmp);

    for (model_layout.TENSORS) |tensor| {
        const before = original[tensor.f32_offset..][0..tensor.len];
        const after = reordered[tensor.f32_offset..][0..tensor.len];
        switch (tensor.layout) {
            .linear, .bias => try testing.expectEqualSlices(f32, before, after),
            .conv3x3, .conv1x1 => {
                const outputs = tensor.shape[0];
                const taps = tensor.shape[1] * tensor.shape[2];
                const inputs = tensor.shape[3];
                for (0..outputs) |output| {
                    for (0..taps) |tap| {
                        for (0..inputs) |input| {
                            try testing.expectEqual(
                                before[(output * taps + tap) * inputs + input],
                                after[(tap * inputs + input) * outputs + output],
                            );
                        }
                    }
                }
            },
        }
    }
}

test "load fills the static weights and marks them loaded" {
    @memcpy(&buffers.weights_bin, go_model_bin);
    loaded = false;
    load();
    try testing.expect(loaded);

    const expected = try testing.allocator.alloc(f32, model_layout.WEIGHTS_F32_LEN);
    defer testing.allocator.free(expected);
    var tmp: [buffers.REORDER_TMP_LEN]f32 = undefined;
    dequantize(go_model_bin, expected);
    reorder(expected, &tmp);
    try testing.expectEqualSlices(u32, @ptrCast(expected), @ptrCast(&buffers.weights_f32));
}
