const std = @import("std");

const WASM_PAGE_BYTES = 64 * 1024;
const WASM_STACK_BYTES = 64 * 1024;

const Model = struct {
    id: []const u8,
    directory: []const u8,
    memory_pages: u32,
    search: SearchCapacity,
};

const SearchCapacity = struct {
    nodes: u32,
    edges: u32,
    memory_pages: u32,
};

const MODELS = [_]Model{
    .{
        .id = "moka-9",
        .directory = "vendor/moka/model",
        .memory_pages = 13,
        .search = .{ .nodes = 8192, .edges = 524_288, .memory_pages = 332 },
    },
    .{
        .id = "moka-9-v2",
        .directory = "vendor/moka-v2/model",
        .memory_pages = 27,
        .search = .{ .nodes = 8192, .edges = 524_288, .memory_pages = 346 },
    },
    .{
        .id = "synthetic-19",
        .directory = "models/synthetic-19",
        .memory_pages = 66,
        .search = .{ .nodes = 4096, .edges = 819_200, .memory_pages = 580 },
    },
    .{
        .id = "synthetic-19-v2",
        .directory = "models/synthetic-19-v2",
        .memory_pages = 81,
        .search = .{ .nodes = 4096, .edges = 819_200, .memory_pages = 581 },
    },
};

const WasmTargets = struct {
    target: std.Build.ResolvedTarget,
    wasm: *std.Build.Step,
    wasm_safe: *std.Build.Step,
    wasm_profile: *std.Build.Step,
    search: *std.Build.Step,
};

pub fn build(b: *std.Build) void {
    const host = b.graph.host;
    const fixtures = b.option([]const u8, "fixtures", "Reference data root directory (default: oracle/fixtures)") orelse "oracle/fixtures";
    const selected = b.option([]const u8, "model", "Build only this model");

    const gen_tool = b.addExecutable(.{
        .name = "gen_model_layout",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/gen_model_layout.zig"),
            .target = host,
            .optimize = .Debug,
        }),
    });

    const wasm_target = b.resolveTargetQuery(.{
        .cpu_arch = .wasm32,
        .os_tag = .freestanding,
        .cpu_features_add = std.Target.wasm.featureSet(&.{.simd128}),
    });

    const gen_step = b.step("gen", "Generate gen/<id>/model_layout.zig from each model manifest");
    const models_step = b.step("models", "Install each model manifest and weights in zig-out/models/<id>/");
    const targets: WasmTargets = .{
        .target = wasm_target,
        .wasm = b.step("wasm", "Build zig-out/<id>/infer.wasm"),
        .wasm_safe = b.step("wasm-safe", "Build zig-out/<id>/infer-safe.wasm (ReleaseSafe)"),
        .wasm_profile = b.step("wasm-profile", "Build zig-out/<id>/infer-profile.wasm, which also exports forward_trunk"),
        .search = b.step("search", "Build zig-out/<id>/search.wasm: inference, Moka's rules and PUCT search"),
    };
    const test_step = b.step("test", "Run the native Zig tests");

    const host_geometry = b.createModule(.{
        .root_source_file = b.path("src/geometry.zig"),
        .target = host,
    });
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = host_geometry })).step);

    const moka = MODELS[0];
    const tool_test_module = b.createModule(.{
        .root_source_file = b.path("tools/gen_model_layout.zig"),
        .target = host,
    });
    tool_test_module.addAnonymousImport("go-model.json", .{ .root_source_file = modelFile(b, moka, "go-model.json") });
    tool_test_module.addAnonymousImport("go-model.bin", .{ .root_source_file = modelFile(b, moka, "go-model.bin") });
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = tool_test_module })).step);

    const models = selectModels(b, selected);
    for (models) |model| {
        const manifest = modelFile(b, model, "go-model.json");
        const weights = modelFile(b, model, "go-model.bin");

        const gen_run = b.addRunArtifact(gen_tool);
        gen_run.addFileArg(manifest);
        gen_run.addFileArg(weights);
        const model_layout_source = gen_run.addOutputFileArg("model_layout.zig");
        gen_step.dependOn(&b.addInstallFile(model_layout_source, b.fmt("gen/{s}/model_layout.zig", .{model.id})).step);

        models_step.dependOn(&b.addInstallFile(manifest, b.fmt("models/{s}/go-model.json", .{model.id})).step);
        models_step.dependOn(&b.addInstallFile(weights, b.fmt("models/{s}/go-model.bin", .{model.id})).step);

        const memory_options = b.addOptions();
        memory_options.addOption(u32, "memory_pages", model.memory_pages);
        memory_options.addOption(u32, "stack_bytes", WASM_STACK_BYTES);
        const memory_config = memory_options.createModule();

        targets.wasm.dependOn(&addInferWasm(b, model, "infer", "src/infer/exports.zig", .ReleaseFast, targets.target, model_layout_source, memory_config, model.memory_pages).step);
        targets.wasm_safe.dependOn(&addInferWasm(b, model, "infer-safe", "src/infer/exports.zig", .ReleaseSafe, targets.target, model_layout_source, memory_config, model.memory_pages).step);
        targets.wasm_profile.dependOn(&addInferWasm(b, model, "infer-profile", "src/infer/profile_exports.zig", .ReleaseFast, targets.target, model_layout_source, memory_config, model.memory_pages).step);

        const search_memory_options = b.addOptions();
        search_memory_options.addOption(u32, "memory_pages", model.search.memory_pages);
        search_memory_options.addOption(u32, "stack_bytes", WASM_STACK_BYTES);
        const search_options = b.addOptions();
        search_options.addOption(u32, "nodes", model.search.nodes);
        search_options.addOption(u32, "edges", model.search.edges);
        const search_wasm = addInferWasm(b, model, "search", "src/search_exports.zig", .ReleaseFast, targets.target, model_layout_source, search_memory_options.createModule(), model.search.memory_pages);
        search_wasm.artifact.root_module.addImport("search_config", search_options.createModule());
        targets.search.dependOn(&search_wasm.step);

        const test_options = b.addOptions();
        const fixtures_root = if (std.fs.path.isAbsolute(fixtures)) b.graph.cwdRelativePath(fixtures) else b.path(fixtures);
        test_options.addOptionPathUntracked("fixtures_dir", fixtures_root.path(b, model.id));
        test_options.addOption([]const u8, "model_id", model.id);

        const src_test_module = b.createModule(.{
            .root_source_file = b.path("src/test_root.zig"),
            .target = host,
            .imports = &.{
                .{ .name = "model_layout", .module = b.createModule(.{
                    .root_source_file = model_layout_source,
                    .target = host,
                }) },
                .{ .name = "geometry", .module = host_geometry },
                .{ .name = "memory_config", .module = memory_options.createModule() },
                .{ .name = "build_options", .module = test_options.createModule() },
            },
        });
        src_test_module.addAnonymousImport("go_model_bin", .{ .root_source_file = weights });
        const src_tests = b.addTest(.{ .root_module = src_test_module, .name = b.fmt("src-{s}", .{model.id}) });
        test_step.dependOn(&b.addRunArtifact(src_tests).step);
    }
}

fn selectModels(b: *std.Build, selected: ?[]const u8) []const Model {
    const id = selected orelse return &MODELS;
    for (&MODELS) |*model| {
        if (std.mem.eql(u8, model.id, id)) return model[0..1];
    }
    var ids: [MODELS.len][]const u8 = undefined;
    for (MODELS, &ids) |model, *known| known.* = model.id;
    const known = std.mem.join(b.allocator, ", ", &ids) catch @panic("OOM");
    std.debug.print("unknown model {s}; known models: {s}\n", .{ id, known });
    std.process.exit(1);
}

fn modelFile(b: *std.Build, model: Model, name: []const u8) std.Build.LazyPath {
    return b.path(b.fmt("{s}/{s}", .{ model.directory, name }));
}

fn addInferWasm(
    b: *std.Build,
    model: Model,
    name: []const u8,
    root: []const u8,
    optimize: std.builtin.OptimizeMode,
    target: std.Build.ResolvedTarget,
    model_layout_source: std.Build.LazyPath,
    memory_config: *std.Build.Module,
    memory_pages: u32,
) *std.Build.Step.InstallArtifact {
    const wasm = b.addExecutable(.{
        .name = name,
        .root_module = b.createModule(.{
            .root_source_file = b.path(root),
            .target = target,
            .optimize = optimize,
            .strip = true,
            .imports = &.{
                .{ .name = "model_layout", .module = b.createModule(.{
                    .root_source_file = model_layout_source,
                    .target = target,
                    .optimize = optimize,
                }) },
                .{ .name = "geometry", .module = b.createModule(.{
                    .root_source_file = b.path("src/geometry.zig"),
                    .target = target,
                    .optimize = optimize,
                }) },
                .{ .name = "memory_config", .module = memory_config },
            },
        }),
    });
    wasm.entry = .disabled;
    wasm.rdynamic = true;
    wasm.import_memory = false;
    wasm.export_memory = true;
    wasm.initial_memory = memory_pages * WASM_PAGE_BYTES;
    wasm.max_memory = memory_pages * WASM_PAGE_BYTES;
    wasm.stack_size = WASM_STACK_BYTES;
    return b.addInstallArtifact(wasm, .{ .dest_dir = .{ .override = .{ .custom = model.id } } });
}
