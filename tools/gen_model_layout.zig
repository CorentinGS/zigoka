const std = @import("std");
const Allocator = std.mem.Allocator;
const json = std.json;

pub const ModelConfig = struct {
    board_size: u16,
    input_planes: u16,
    trunk_channels: u16,
    bottleneck_channels: u16,
    residual_blocks: u16,
    global_block_interval: u16,
    global_hidden_channels: u16,
    policy_channels: u16,
    policy_moves: u16,
    value_channels: u16,
    value_hidden_channels: u16,
};

const config_json_keys = [_]struct { []const u8, []const u8 }{
    .{ "board_size", "boardSize" },
    .{ "input_planes", "inputPlaneCount" },
    .{ "trunk_channels", "trunkChannelCount" },
    .{ "bottleneck_channels", "bottleneckChannelCount" },
    .{ "residual_blocks", "residualBlockCount" },
    .{ "global_block_interval", "globalResidualBlockInterval" },
    .{ "global_hidden_channels", "globalResidualHiddenChannelCount" },
    .{ "policy_channels", "policyChannelCount" },
    .{ "policy_moves", "policyMoveCount" },
    .{ "value_channels", "valueChannelCount" },
    .{ "value_hidden_channels", "scoreHiddenChannelCount" },
};

const config_field_names = @typeInfo(ModelConfig).@"struct".field_names;

comptime {
    if (config_field_names.len != config_json_keys.len) @compileError("config_json_keys does not cover ModelConfig");
    for (config_field_names, config_json_keys) |field_name, entry| {
        if (!std.mem.eql(u8, field_name, entry[0])) @compileError("config_json_keys is out of order");
    }
    for (@typeInfo(ModelConfig).@"struct".field_types) |field_type| {
        if (field_type != u16) @compileError("render assumes every ModelConfig field is a u16");
    }
}

pub const Kind = enum { int8, float32 };
pub const Layout = enum { conv3x3, conv1x1, linear, bias };

pub const Tensor = struct {
    name: []const u8,
    kind: Kind,
    layout: Layout,
    shape: []const u32,
    data_offset: u32,
    scale_offset: ?u32,
    f32_offset: u32,
    len: u32,
};

pub const Model = struct {
    config: ModelConfig,
    tensors: []const Tensor,
    weights_bytes: u32,
    weights_f32_len: u32,
    sha256: [32]u8,
};

pub const Error = error{
    InvalidManifest,
    UnsupportedFormat,
    UnsupportedVersion,
    UnsupportedBlockKind,
    InvalidArchitecture,
    UnsupportedDtype,
    UnsupportedQuantizationGroup,
    MissingScale,
    OutOfBounds,
    UnexpectedTensor,
    MissingTensor,
    ShapeMismatch,
    WeightsSizeMismatch,
    Sha256Mismatch,
    OutOfMemory,
};

pub const Diagnostic = struct {
    message: []const u8 = "",
};

const supported_format = "million-go-int8";
const supported_version = 1;
const supported_block_kind = "nested-bottleneck";

fn fail(arena: Allocator, diagnostic: *Diagnostic, err: Error, comptime format: []const u8, args: anytype) Error {
    diagnostic.message = try std.fmt.allocPrint(arena, format, args);
    return err;
}

fn getField(object: json.ObjectMap, key: []const u8) ?json.Value {
    return object.get(key);
}

fn getObject(value: ?json.Value) ?json.ObjectMap {
    const present = value orelse return null;
    return switch (present) {
        .object => |object| object,
        else => null,
    };
}

fn getString(value: ?json.Value) ?[]const u8 {
    const present = value orelse return null;
    return switch (present) {
        .string => |string| string,
        else => null,
    };
}

fn getInteger(value: ?json.Value) ?i64 {
    const present = value orelse return null;
    return switch (present) {
        .integer => |integer| integer,
        else => null,
    };
}

fn getU32(value: ?json.Value) ?u32 {
    const integer = getInteger(value) orelse return null;
    return std.math.cast(u32, integer);
}

const Required = struct {
    name: []const u8,
    shape: []const u32,
};

fn requiredTensors(arena: Allocator, config: ModelConfig) Allocator.Error![]const Required {
    const area: u32 = @as(u32, config.board_size) * config.board_size;
    const trunk: u32 = config.trunk_channels;
    const bottleneck: u32 = config.bottleneck_channels;
    const global_hidden: u32 = config.global_hidden_channels;

    var list: std.ArrayList(Required) = .empty;
    const add = struct {
        fn add(a: Allocator, l: *std.ArrayList(Required), name: []const u8, shape: []const u32) Allocator.Error!void {
            try l.append(a, .{ .name = name, .shape = try a.dupe(u32, shape) });
        }
    }.add;

    try add(arena, &list, "stem.weight", &.{ trunk, 3, 3, config.input_planes });
    try add(arena, &list, "stem.bias", &.{trunk});
    for (0..config.residual_blocks) |block| {
        const prefix = try std.fmt.allocPrint(arena, "residual.{d}", .{block});
        const name = struct {
            fn name(a: Allocator, p: []const u8, suffix: []const u8) Allocator.Error![]const u8 {
                return std.fmt.allocPrint(a, "{s}.{s}", .{ p, suffix });
            }
        }.name;
        try add(arena, &list, try name(arena, prefix, "reduce.weight"), &.{ bottleneck, 1, 1, trunk });
        try add(arena, &list, try name(arena, prefix, "reduce.bias"), &.{bottleneck});
        try add(arena, &list, try name(arena, prefix, "first.weight"), &.{ bottleneck, 3, 3, bottleneck });
        try add(arena, &list, try name(arena, prefix, "first.bias"), &.{bottleneck});
        if (isGlobalBlock(config, block)) {
            try add(arena, &list, try name(arena, prefix, "global.hidden.weight"), &.{ global_hidden, 2 * bottleneck });
            try add(arena, &list, try name(arena, prefix, "global.hidden.bias"), &.{global_hidden});
            try add(arena, &list, try name(arena, prefix, "global.output.weight"), &.{ bottleneck, global_hidden });
            try add(arena, &list, try name(arena, prefix, "global.output.bias"), &.{bottleneck});
        }
        try add(arena, &list, try name(arena, prefix, "second.weight"), &.{ bottleneck, 3, 3, bottleneck });
        try add(arena, &list, try name(arena, prefix, "second.bias"), &.{bottleneck});
        try add(arena, &list, try name(arena, prefix, "expand.weight"), &.{ trunk, 1, 1, bottleneck });
        try add(arena, &list, try name(arena, prefix, "expand.bias"), &.{trunk});
    }
    try add(arena, &list, "policy.convolution.weight", &.{ config.policy_channels, 1, 1, trunk });
    try add(arena, &list, "policy.convolution.bias", &.{config.policy_channels});
    try add(arena, &list, "policy.linear.weight", &.{ config.policy_moves, config.policy_channels * area });
    try add(arena, &list, "policy.linear.bias", &.{config.policy_moves});
    try add(arena, &list, "value.convolution.weight", &.{ config.value_channels, 1, 1, trunk });
    try add(arena, &list, "value.convolution.bias", &.{config.value_channels});
    try add(arena, &list, "value.hidden.weight", &.{ config.value_hidden_channels, config.value_channels * area });
    try add(arena, &list, "value.hidden.bias", &.{config.value_hidden_channels});
    try add(arena, &list, "value.output.weight", &.{ 1, config.value_hidden_channels });
    try add(arena, &list, "value.output.bias", &.{1});
    return list.items;
}

pub fn isGlobalBlock(config: ModelConfig, block: usize) bool {
    return config.global_block_interval > 0 and (block + 1) % config.global_block_interval == 0;
}

fn layoutOf(shape: []const u32) Layout {
    return switch (shape.len) {
        1 => .bias,
        2 => .linear,
        else => if (shape[1] == 3) .conv3x3 else .conv1x1,
    };
}

fn parseConfig(arena: Allocator, diagnostic: *Diagnostic, architecture: json.ObjectMap) Error!ModelConfig {
    var config: ModelConfig = undefined;
    inline for (config_field_names, config_json_keys) |field_name, entry| {
        const integer = getInteger(getField(architecture, entry[1])) orelse
            return fail(arena, diagnostic, error.InvalidArchitecture, "architecture.{s} is missing or not an integer", .{entry[1]});
        @field(config, field_name) = std.math.cast(u16, integer) orelse
            return fail(arena, diagnostic, error.InvalidArchitecture, "architecture.{s} is out of range: {d}", .{ entry[1], integer });
    }
    const block_kind = getString(getField(architecture, "residualBlockKind")) orelse "";
    if (!std.mem.eql(u8, block_kind, supported_block_kind)) {
        return fail(arena, diagnostic, error.UnsupportedBlockKind, "residualBlockKind must be {s}, found \"{s}\"", .{ supported_block_kind, block_kind });
    }
    if (config.board_size == 0 or config.input_planes == 0 or config.trunk_channels == 0 or
        config.bottleneck_channels == 0 or config.policy_channels == 0 or config.policy_moves == 0 or
        config.value_channels == 0 or config.value_hidden_channels == 0)
    {
        return fail(arena, diagnostic, error.InvalidArchitecture, "architecture has a zero channel or board size", .{});
    }
    if ((config.global_block_interval == 0) != (config.global_hidden_channels == 0)) {
        return fail(arena, diagnostic, error.InvalidArchitecture, "globalResidualBlockInterval and globalResidualHiddenChannelCount must both be zero or both be positive", .{});
    }
    return config;
}

pub fn analyze(arena: Allocator, manifest: json.Value, weights: []const u8, diagnostic: *Diagnostic) Error!Model {
    const root = getObject(manifest) orelse
        return fail(arena, diagnostic, error.InvalidManifest, "the manifest is not a JSON object", .{});

    const format = getString(getField(root, "format")) orelse "";
    if (!std.mem.eql(u8, format, supported_format)) {
        return fail(arena, diagnostic, error.UnsupportedFormat, "format must be {s}, found \"{s}\"", .{ supported_format, format });
    }
    const version = getInteger(getField(root, "version")) orelse -1;
    if (version != supported_version) {
        return fail(arena, diagnostic, error.UnsupportedVersion, "version must be {d}, found {d}", .{ supported_version, version });
    }
    const architecture = getObject(getField(root, "architecture")) orelse
        return fail(arena, diagnostic, error.InvalidManifest, "architecture is missing", .{});
    const config = try parseConfig(arena, diagnostic, architecture);

    const weights_bytes = getU32(getField(root, "weightsBytes")) orelse
        return fail(arena, diagnostic, error.InvalidManifest, "weightsBytes is missing or out of range", .{});
    const sha256_hex = getString(getField(root, "sha256")) orelse "";
    var sha256: [32]u8 = undefined;
    if (sha256_hex.len != 64) {
        return fail(arena, diagnostic, error.InvalidManifest, "sha256 must have 64 hexadecimal digits", .{});
    }
    _ = std.fmt.hexToBytes(&sha256, sha256_hex) catch
        return fail(arena, diagnostic, error.InvalidManifest, "sha256 is not hexadecimal", .{});

    const manifest_tensors = getObject(getField(root, "tensors")) orelse
        return fail(arena, diagnostic, error.InvalidManifest, "tensors is missing", .{});

    var iterator = manifest_tensors.iterator();
    while (iterator.next()) |entry| {
        const name = entry.key_ptr.*;
        const tensor = getObject(entry.value_ptr.*) orelse
            return fail(arena, diagnostic, error.InvalidManifest, "tensor {s} is not an object", .{name});
        const dtype = getString(getField(tensor, "dtype")) orelse "";
        if (!std.mem.eql(u8, dtype, "int8") and !std.mem.eql(u8, dtype, "float32")) {
            return fail(arena, diagnostic, error.UnsupportedDtype, "tensor {s} has unsupported dtype \"{s}\"", .{ name, dtype });
        }
        if (getField(tensor, "quantizationGroupSize") != null) {
            return fail(arena, diagnostic, error.UnsupportedQuantizationGroup, "tensor {s} uses quantizationGroupSize", .{name});
        }
        if (std.mem.eql(u8, dtype, "int8") and getField(tensor, "scaleOffset") == null) {
            return fail(arena, diagnostic, error.MissingScale, "Missing quantization scale for {s}.", .{name});
        }
    }

    const required = try requiredTensors(arena, config);
    const tensors = try arena.alloc(Tensor, required.len);
    var f32_offset: u32 = 0;
    for (required, tensors) |want, *out| {
        const tensor = getObject(manifest_tensors.get(want.name)) orelse
            return fail(arena, diagnostic, error.MissingTensor, "Model tensor {s} is missing.", .{want.name});

        const shape_value = getField(tensor, "shape") orelse json.Value.null;
        const shape_items = switch (shape_value) {
            .array => |array| array.items,
            else => return fail(arena, diagnostic, error.InvalidManifest, "tensor {s} has no shape", .{want.name}),
        };
        var shape_matches = shape_items.len == want.shape.len;
        if (shape_matches) {
            for (shape_items, want.shape) |item, dimension| {
                if (getU32(item) != dimension) shape_matches = false;
            }
        }
        if (!shape_matches) {
            return fail(arena, diagnostic, error.ShapeMismatch, "tensor {s} must have shape {any}", .{ want.name, want.shape });
        }

        var len: u32 = 1;
        for (want.shape) |dimension| len *= dimension;

        const kind: Kind = if (std.mem.eql(u8, getString(getField(tensor, "dtype")).?, "int8")) .int8 else .float32;
        const data_offset = getU32(getField(tensor, "dataOffset")) orelse
            return fail(arena, diagnostic, error.InvalidManifest, "tensor {s} has no valid dataOffset", .{want.name});
        const data_bytes: u64 = if (kind == .int8) len else @as(u64, len) * 4;
        if (data_offset + data_bytes > weights_bytes) {
            return fail(arena, diagnostic, error.OutOfBounds, "tensor {s} data ends past weightsBytes", .{want.name});
        }
        var scale_offset: ?u32 = null;
        if (kind == .int8) {
            const offset = getU32(getField(tensor, "scaleOffset")) orelse
                return fail(arena, diagnostic, error.InvalidManifest, "tensor {s} has no valid scaleOffset", .{want.name});
            if (offset + @as(u64, want.shape[0]) * 4 > weights_bytes) {
                return fail(arena, diagnostic, error.OutOfBounds, "tensor {s} scales end past weightsBytes", .{want.name});
            }
            scale_offset = offset;
        }

        out.* = .{
            .name = want.name,
            .kind = kind,
            .layout = layoutOf(want.shape),
            .shape = want.shape,
            .data_offset = data_offset,
            .scale_offset = scale_offset,
            .f32_offset = f32_offset,
            .len = len,
        };
        f32_offset += len;
    }

    if (manifest_tensors.count() != required.len) {
        var names = manifest_tensors.iterator();
        while (names.next()) |entry| {
            const known = for (required) |want| {
                if (std.mem.eql(u8, want.name, entry.key_ptr.*)) break true;
            } else false;
            if (!known) {
                return fail(arena, diagnostic, error.UnexpectedTensor, "tensor {s} is not part of this architecture", .{entry.key_ptr.*});
            }
        }
    }

    if (weights.len != weights_bytes) {
        return fail(arena, diagnostic, error.WeightsSizeMismatch, "the weights file has {d} bytes, the manifest expects {d}", .{ weights.len, weights_bytes });
    }
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(weights, &digest, .{});
    if (!std.mem.eql(u8, &digest, &sha256)) {
        return fail(arena, diagnostic, error.Sha256Mismatch, "the weights file sha256 is {x}, the manifest expects {s}", .{ &digest, sha256_hex });
    }

    return .{
        .config = config,
        .tensors = tensors,
        .weights_bytes = weights_bytes,
        .weights_f32_len = f32_offset,
        .sha256 = sha256,
    };
}

pub fn render(writer: *std.Io.Writer, model: Model) std.Io.Writer.Error!void {
    try writer.writeAll(
        \\// Generated by tools/gen_model_layout.zig from the Moka manifest. Do not edit.
        \\const std = @import("std");
        \\
        \\pub const ModelConfig = struct {
        \\
    );
    inline for (config_field_names) |field_name| {
        try writer.print("    {s}: u16,\n", .{field_name});
    }
    try writer.writeAll(
        \\};
        \\pub const Kind = enum { int8, float32 };
        \\pub const Layout = enum { conv3x3, conv1x1, linear, bias };
        \\pub const Tensor = struct {
        \\    name: []const u8,
        \\    kind: Kind,
        \\    layout: Layout,
        \\    shape: []const u32,
        \\    data_offset: u32,
        \\    scale_offset: ?u32,
        \\    f32_offset: u32,
        \\    len: u32,
        \\};
        \\
        \\pub const CONFIG: ModelConfig = .{
        \\
    );
    inline for (config_field_names) |field_name| {
        try writer.print("    .{s} = {d},\n", .{ field_name, @field(model.config, field_name) });
    }
    try writer.writeAll("};\n\npub const TENSORS = [_]Tensor{\n");
    for (model.tensors) |tensor| {
        try writer.print("    .{{ .name = \"{s}\", .kind = .{s}, .layout = .{s}, .shape = &.{{", .{
            tensor.name, @tagName(tensor.kind), @tagName(tensor.layout),
        });
        for (tensor.shape, 0..) |dimension, index| {
            try writer.print("{s}{d}", .{ if (index == 0) " " else ", ", dimension });
        }
        try writer.print(" }}, .data_offset = {d}, .scale_offset = ", .{tensor.data_offset});
        if (tensor.scale_offset) |offset| {
            try writer.print("{d}", .{offset});
        } else {
            try writer.writeAll("null");
        }
        try writer.print(", .f32_offset = {d}, .len = {d} }},\n", .{ tensor.f32_offset, tensor.len });
    }
    try writer.print(
        \\}};
        \\
        \\pub const WEIGHTS_BYTES: u32 = {d};
        \\pub const WEIGHTS_F32_LEN: u32 = {d};
        \\pub const SHA256: [32]u8 = .{{
    , .{ model.weights_bytes, model.weights_f32_len });
    for (model.sha256, 0..) |byte, index| {
        try writer.print("{s}0x{x:0>2}", .{ if (index == 0) " " else ", ", byte });
    }
    try writer.writeAll(
        \\ };
        \\
        \\pub fn tensor(comptime name: []const u8) Tensor {
        \\    inline for (TENSORS) |candidate| {
        \\        if (comptime std.mem.eql(u8, candidate.name, name)) return candidate;
        \\    }
        \\    @compileError("model tensor " ++ name ++ " is missing");
        \\}
        \\
    );
}

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(arena);
    if (args.len != 4) {
        std.debug.print("usage: gen_model_layout <go-model.json> <go-model.bin> <model_layout.zig>\n", .{});
        std.process.exit(2);
    }

    const cwd = std.Io.Dir.cwd();
    const manifest_text = try cwd.readFileAlloc(io, args[1], arena, .limited(16 * 1024 * 1024));
    const weights = try cwd.readFileAlloc(io, args[2], arena, .limited(256 * 1024 * 1024));
    const manifest = json.parseFromSliceLeaky(json.Value, arena, manifest_text, .{}) catch |err| {
        std.debug.print("{s}: invalid JSON: {t}\n", .{ args[1], err });
        std.process.exit(1);
    };

    var diagnostic: Diagnostic = .{};
    const model = analyze(arena, manifest, weights, &diagnostic) catch |err| {
        std.debug.print("{s}: {t}: {s}\n", .{ args[1], err, diagnostic.message });
        std.process.exit(1);
    };

    var output: std.Io.Writer.Allocating = .init(arena);
    try render(&output.writer, model);
    try cwd.writeFile(io, .{ .sub_path = args[3], .data = output.written() });
}

const testing = std.testing;
const real_manifest_text = @embedFile("go-model.json");
const real_weights = @embedFile("go-model.bin");

fn parseReal(arena: Allocator) !json.Value {
    return json.parseFromSliceLeaky(json.Value, arena, real_manifest_text, .{});
}

fn tensorObject(manifest: *json.Value, name: []const u8) *json.ObjectMap {
    return &manifest.object.getPtr("tensors").?.object.getPtr(name).?.object;
}

test "the Moka manifest gives the expected layout" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var diagnostic: Diagnostic = .{};
    const model = try analyze(arena, try parseReal(arena), real_weights, &diagnostic);
    try testing.expectEqual(@as(usize, 120), model.tensors.len);
    try testing.expectEqual(@as(u32, 105_353), model.weights_f32_len);
    try testing.expectEqual(@as(u32, 113_648), model.weights_bytes);
    try testing.expectEqual(@as(u16, 9), model.config.board_size);
    try testing.expectEqual(@as(u16, 32), model.config.value_hidden_channels);
    try testing.expectEqualStrings("stem.weight", model.tensors[0].name);
    try testing.expectEqual(Layout.conv3x3, model.tensors[0].layout);
    try testing.expectEqualStrings("value.output.bias", model.tensors[model.tensors.len - 1].name);
    try testing.expectEqual(Layout.bias, model.tensors[model.tensors.len - 1].layout);

    var output: std.Io.Writer.Allocating = .init(arena);
    try render(&output.writer, model);
    try testing.expect(std.mem.indexOf(u8, output.written(), "pub const WEIGHTS_F32_LEN: u32 = 105353;") != null);
}

fn expectRejected(expected: Error, mutate: *const fn (Allocator, *json.Value) anyerror!void) !void {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var manifest = try parseReal(arena);
    try mutate(arena, &manifest);
    var diagnostic: Diagnostic = .{};
    try testing.expectError(expected, analyze(arena, manifest, real_weights, &diagnostic));
    try testing.expect(diagnostic.message.len > 0);
}

test "an int4 tensor is rejected" {
    try expectRejected(error.UnsupportedDtype, struct {
        fn mutate(arena: Allocator, manifest: *json.Value) !void {
            try tensorObject(manifest, "residual.4.first.weight").put(arena, "dtype", .{ .string = "int4" });
        }
    }.mutate);
}

test "a tensor with quantizationGroupSize is rejected" {
    try expectRejected(error.UnsupportedQuantizationGroup, struct {
        fn mutate(arena: Allocator, manifest: *json.Value) !void {
            try tensorObject(manifest, "policy.linear.weight").put(arena, "quantizationGroupSize", .{ .integer = 81 });
        }
    }.mutate);
}

test "an int8 tensor without scale is rejected" {
    try expectRejected(error.MissingScale, struct {
        fn mutate(_: Allocator, manifest: *json.Value) !void {
            _ = tensorObject(manifest, "stem.weight").orderedRemove("scaleOffset");
        }
    }.mutate);
}

test "a missing tensor is rejected" {
    try expectRejected(error.MissingTensor, struct {
        fn mutate(_: Allocator, manifest: *json.Value) !void {
            _ = manifest.object.getPtr("tensors").?.object.orderedRemove("residual.7.global.output.bias");
        }
    }.mutate);
}

test "an unexpected tensor is rejected" {
    try expectRejected(error.UnexpectedTensor, struct {
        fn mutate(arena: Allocator, manifest: *json.Value) !void {
            const tensors = &manifest.object.getPtr("tensors").?.object;
            try tensors.put(arena, "residual.0.global.hidden.bias", tensors.get("residual.3.global.hidden.bias").?);
        }
    }.mutate);
}

test "a wrong shape is rejected" {
    try expectRejected(error.ShapeMismatch, struct {
        fn mutate(_: Allocator, manifest: *json.Value) !void {
            tensorObject(manifest, "value.hidden.weight").getPtr("shape").?.array.items[1] = .{ .integer = 161 };
        }
    }.mutate);
}

test "an offset past weightsBytes is rejected" {
    try expectRejected(error.OutOfBounds, struct {
        fn mutate(arena: Allocator, manifest: *json.Value) !void {
            try tensorObject(manifest, "value.output.bias").put(arena, "dataOffset", .{ .integer = 113_646 });
        }
    }.mutate);
}

test "another format or block kind is rejected" {
    try expectRejected(error.UnsupportedFormat, struct {
        fn mutate(arena: Allocator, manifest: *json.Value) !void {
            try manifest.object.put(arena, "format", .{ .string = "million-go-int4" });
        }
    }.mutate);
    try expectRejected(error.UnsupportedBlockKind, struct {
        fn mutate(arena: Allocator, manifest: *json.Value) !void {
            try manifest.object.getPtr("architecture").?.object.put(arena, "residualBlockKind", .{ .string = "standard" });
        }
    }.mutate);
}

test "modified or truncated weights are rejected" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const manifest = try parseReal(arena);

    var diagnostic: Diagnostic = .{};
    const modified = try arena.dupe(u8, real_weights);
    modified[1000] +%= 1;
    try testing.expectError(error.Sha256Mismatch, analyze(arena, manifest, modified, &diagnostic));
    try testing.expectError(error.WeightsSizeMismatch, analyze(arena, manifest, real_weights[0 .. real_weights.len - 1], &diagnostic));
}
