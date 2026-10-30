const std = @import("std");

const WASM_PAGE_BYTES = 64 * 1024;
const WASM_MEMORY_PAGES = 13;
const WASM_STACK_BYTES = 64 * 1024;

pub fn build(b: *std.Build) void {
    const host = b.graph.host;
    const fixtures = b.option([]const u8, "fixtures", "Reference data directory (default: oracle/fixtures)") orelse "oracle/fixtures";

    const manifest = b.path("vendor/moka/model/go-model.json");
    const weights = b.path("vendor/moka/model/go-model.bin");

    const gen_tool = b.addExecutable(.{
        .name = "gen_model_layout",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/gen_model_layout.zig"),
            .target = host,
            .optimize = .Debug,
        }),
    });
    const gen_run = b.addRunArtifact(gen_tool);
    gen_run.addFileArg(manifest);
    gen_run.addFileArg(weights);
    const model_layout_source = gen_run.addOutputFileArg("model_layout.zig");

    const gen_step = b.step("gen", "Generate model_layout.zig from the Moka manifest");
    gen_step.dependOn(&b.addInstallFile(model_layout_source, "gen/model_layout.zig").step);

    const memory_options = b.addOptions();
    memory_options.addOption(u32, "memory_pages", WASM_MEMORY_PAGES);
    memory_options.addOption(u32, "stack_bytes", WASM_STACK_BYTES);

    const wasm_target = b.resolveTargetQuery(.{
        .cpu_arch = .wasm32,
        .os_tag = .freestanding,
        .cpu_features_add = std.Target.wasm.featureSet(&.{.simd128}),
    });
    const wasm_model_layout = b.createModule(.{
        .root_source_file = model_layout_source,
        .target = wasm_target,
        .optimize = .ReleaseFast,
    });
    const wasm_geometry = b.createModule(.{
        .root_source_file = b.path("src/geometry.zig"),
        .target = wasm_target,
        .optimize = .ReleaseFast,
    });
    const wasm = b.addExecutable(.{
        .name = "infer",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/infer/exports.zig"),
            .target = wasm_target,
            .optimize = .ReleaseFast,
            .strip = true,
            .imports = &.{
                .{ .name = "model_layout", .module = wasm_model_layout },
                .{ .name = "geometry", .module = wasm_geometry },
                .{ .name = "memory_config", .module = memory_options.createModule() },
            },
        }),
    });
    wasm.entry = .disabled;
    wasm.rdynamic = true;
    wasm.import_memory = false;
    wasm.export_memory = true;
    wasm.initial_memory = WASM_MEMORY_PAGES * WASM_PAGE_BYTES;
    wasm.max_memory = WASM_MEMORY_PAGES * WASM_PAGE_BYTES;
    wasm.stack_size = WASM_STACK_BYTES;

    const wasm_step = b.step("wasm", "Build zig-out/bin/infer.wasm");
    wasm_step.dependOn(&b.addInstallArtifact(wasm, .{}).step);

    const test_options = b.addOptions();
    const fixtures_path = if (std.fs.path.isAbsolute(fixtures)) b.graph.cwdRelativePath(fixtures) else b.path(fixtures);
    test_options.addOptionPathUntracked("fixtures_dir", fixtures_path);

    const host_model_layout = b.createModule(.{
        .root_source_file = model_layout_source,
        .target = host,
    });
    const host_geometry = b.createModule(.{
        .root_source_file = b.path("src/geometry.zig"),
        .target = host,
    });
    const geometry_tests = b.addTest(.{ .root_module = host_geometry });

    const src_test_module = b.createModule(.{
        .root_source_file = b.path("src/test_root.zig"),
        .target = host,
        .imports = &.{
            .{ .name = "model_layout", .module = host_model_layout },
            .{ .name = "geometry", .module = host_geometry },
            .{ .name = "memory_config", .module = memory_options.createModule() },
            .{ .name = "build_options", .module = test_options.createModule() },
        },
    });
    src_test_module.addAnonymousImport("go_model_bin", .{ .root_source_file = weights });
    const src_tests = b.addTest(.{ .root_module = src_test_module });

    const tool_test_module = b.createModule(.{
        .root_source_file = b.path("tools/gen_model_layout.zig"),
        .target = host,
    });
    tool_test_module.addAnonymousImport("go-model.json", .{ .root_source_file = manifest });
    tool_test_module.addAnonymousImport("go-model.bin", .{ .root_source_file = weights });
    const tool_tests = b.addTest(.{ .root_module = tool_test_module });

    const test_step = b.step("test", "Run the native Zig tests");
    test_step.dependOn(&b.addRunArtifact(geometry_tests).step);
    test_step.dependOn(&b.addRunArtifact(src_tests).step);
    test_step.dependOn(&b.addRunArtifact(tool_tests).step);
}
