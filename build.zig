const std = @import("std");

const WASM_PAGE_BYTES = 64 * 1024;
const WASM_STACK_BYTES = 64 * 1024;

/// The registry of the models the repository builds and tests, relative to the build root.
const DEFAULT_REGISTRY = "models/registry.json";
/// The model whose files the tests of tools/gen_model_layout.zig embed.
const TOOL_TEST_MODEL_ID = "moka-9";

const Model = struct {
    id: []const u8,
    /// The folder of go-model.json and go-model.bin.
    directory: std.Build.LazyPath,
    memory_pages: u32,
    search: SearchCapacity,
    /// A local model (-Dlocal-model) is built under zig-out/local/ and has no fixtures.
    local: bool = false,
};

const SearchCapacity = struct {
    nodes: u32,
    edges: u32,
    memory_pages: u32,
};

/// models/registry.json, with the file's own key names. Every field without a default is
/// required, and an unknown or repeated key is refused.
const Registry = struct {
    models: []const struct {
        id: []const u8,
        /// Relative to the repository root: the parent of the registry's folder.
        directory: []const u8,
        kind: enum { trained, synthetic },
        /// A synthetic model names the trained model whose trunk it keeps.
        trunkFrom: ?[]const u8 = null,
        memoryPages: u32,
        search: struct { nodes: u32, edges: u32, memoryPages: u32 },
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
    const local = b.option([]const u8, "local-model", "Also build models/local/<id>/ (never versioned) with the memory of a registered model of the same architecture: <id>:<registered id>");
    const registry = b.option([]const u8, "registry", "Model registry (default: " ++ DEFAULT_REGISTRY ++ "); the repository root is the parent of its folder") orelse DEFAULT_REGISTRY;
    const registered = loadRegistry(b, registry);

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

    const moka = findRegistered(registered, TOOL_TEST_MODEL_ID) orelse
        fail("{s} has no model {s}, whose files the tests of tools/gen_model_layout.zig embed", .{ registry, TOOL_TEST_MODEL_ID });
    const tool_test_module = b.createModule(.{
        .root_source_file = b.path("tools/gen_model_layout.zig"),
        .target = host,
    });
    tool_test_module.addAnonymousImport("go-model.json", .{ .root_source_file = modelFile(b, moka, "go-model.json") });
    tool_test_module.addAnonymousImport("go-model.bin", .{ .root_source_file = modelFile(b, moka, "go-model.bin") });
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = tool_test_module })).step);

    const models = withLocalModel(b, registry, registered, selectModels(b, registered, selected), local);
    for (models) |model| {
        const manifest = modelFile(b, model, "go-model.json");
        const weights = modelFile(b, model, "go-model.bin");
        const prefix = if (model.local) "local/" else "";

        const gen_run = b.addRunArtifact(gen_tool);
        gen_run.addFileArg(manifest);
        gen_run.addFileArg(weights);
        const model_layout_source = gen_run.addOutputFileArg("model_layout.zig");
        gen_step.dependOn(&b.addInstallFile(model_layout_source, b.fmt("{s}gen/{s}/model_layout.zig", .{ prefix, model.id })).step);

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

        // The native tests compare against oracle/fixtures/<id>/, which a local model does not have.
        if (model.local) continue;

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

fn selectModels(b: *std.Build, registered: []const Model, selected: ?[]const u8) []const Model {
    const id = selected orelse return registered;
    for (registered) |*model| {
        if (std.mem.eql(u8, model.id, id)) return model[0..1];
    }
    const ids = b.allocator.alloc([]const u8, registered.len) catch @panic("OOM");
    for (registered, ids) |model, *known| known.* = model.id;
    const known = std.mem.join(b.allocator, ", ", ids) catch @panic("OOM");
    fail("unknown model {s}; known models: {s}", .{ id, known });
}

fn findRegistered(registered: []const Model, id: []const u8) ?Model {
    for (registered) |model| {
        if (std.mem.eql(u8, model.id, id)) return model;
    }
    return null;
}

/// Reads and checks the model registry. The configure step depends on its contents, so editing
/// it reconfigures the build.
fn loadRegistry(b: *std.Build, registry: []const u8) []const Model {
    const absolute = std.fs.path.isAbsolute(registry);
    b.dependOnFileContents(if (absolute) b.graph.cwdRelativePath(registry) else b.path(registry));

    const file_path = if (absolute) registry else b.root.joinString(b.allocator, registry) catch @panic("OOM");
    const bytes = std.Io.Dir.cwd().readFileAlloc(b.graph.io, file_path, b.allocator, .limited(1 << 20)) catch |err|
        fail("{s}: {s}", .{ registry, @errorName(err) });

    var scanner = std.json.Scanner.initCompleteInput(b.allocator, bytes);
    var diagnostics: std.json.Diagnostics = .{};
    scanner.enableDiagnostics(&diagnostics);
    const parsed = std.json.parseFromTokenSourceLeaky(Registry, b.allocator, &scanner, .{}) catch |err|
        fail("{s}:{d}:{d}: {s}", .{ registry, diagnostics.getLine(), diagnostics.getColumn(), @errorName(err) });

    if (parsed.models.len == 0) fail("{s} lists no model", .{registry});

    const models = b.allocator.alloc(Model, parsed.models.len) catch @panic("OOM");
    for (parsed.models, models, 0..) |entry, *model, index| {
        checkModelId(registry, entry.id);
        for (parsed.models[0..index]) |previous| {
            if (std.mem.eql(u8, previous.id, entry.id)) fail("{s}: model {s} is listed twice", .{ registry, entry.id });
        }
        switch (entry.kind) {
            .trained => if (entry.trunkFrom != null) fail("{s}: {s} is trained and cannot have a trunkFrom", .{ registry, entry.id }),
            .synthetic => {
                const source = entry.trunkFrom orelse fail("{s}: synthetic model {s} needs a trunkFrom", .{ registry, entry.id });
                const trunk_model = for (parsed.models) |candidate| {
                    if (std.mem.eql(u8, candidate.id, source)) break candidate;
                } else fail("{s}: {s} takes its trunk from {s}, which is not registered", .{ registry, entry.id, source });
                if (trunk_model.kind != .trained) fail("{s}: {s} takes its trunk from {s}, which is not a trained model", .{ registry, entry.id, source });
            },
        }
        model.* = .{
            .id = entry.id,
            .directory = fromRepositoryRoot(b, registry, entry.directory),
            .memory_pages = entry.memoryPages,
            .search = .{ .nodes = entry.search.nodes, .edges = entry.search.edges, .memory_pages = entry.search.memoryPages },
        };
    }
    return models;
}

/// Model ids name folders: under zig-out/, oracle/fixtures/ and models/local/.
fn checkModelId(registry: []const u8, id: []const u8) void {
    if (id.len == 0 or std.mem.indexOfAny(u8, id, "/\\.") != null) {
        fail("{s}: {s} is not a plain directory name", .{ registry, id });
    }
}

/// A path relative to the repository root, which is the parent of the registry's folder.
fn fromRepositoryRoot(b: *std.Build, registry: []const u8, sub_path: []const u8) std.Build.LazyPath {
    const root = if (std.fs.path.dirname(registry)) |folder| std.fs.path.dirname(folder) else null;
    const joined = if (root) |directory| b.pathJoin(&.{ directory, sub_path }) else sub_path;
    return if (std.fs.path.isAbsolute(joined)) b.graph.cwdRelativePath(joined) else b.path(joined);
}

fn fail(comptime format: []const u8, args: anytype) noreturn {
    std.debug.print(format ++ "\n", args);
    std.process.exit(1);
}

/// Adds a local model: an export of a `pipeline/` checkpoint in models/local/<id>/, which is
/// never versioned, built into zig-out/local/. It borrows the memory and search capacity of the
/// registered model `like`, which must have the same architecture; the manifest checks of
/// gen_model_layout still apply, and the native tests skip it.
fn withLocalModel(b: *std.Build, registry: []const u8, registered: []const Model, models: []const Model, spec: ?[]const u8) []const Model {
    const text = spec orelse return models;
    const colon = std.mem.indexOfScalar(u8, text, ':') orelse fail("-Dlocal-model expects <id>:<registered id>, got {s}", .{text});
    const id = text[0..colon];
    const like_id = text[colon + 1 ..];
    if (id.len == 0 or std.mem.indexOfAny(u8, id, "/\\.") != null) {
        fail("-Dlocal-model: {s} is not a plain directory name", .{id});
    }
    if (findRegistered(registered, id) != null) fail("-Dlocal-model: {s} is already a registered model", .{id});
    const like = selectModels(b, registered, like_id)[0];
    const list = b.allocator.alloc(Model, models.len + 1) catch @panic("OOM");
    @memcpy(list[0..models.len], models);
    list[models.len] = .{
        .id = id,
        .directory = fromRepositoryRoot(b, registry, b.fmt("models/local/{s}", .{id})),
        .memory_pages = like.memory_pages,
        .search = like.search,
        .local = true,
    };
    return list;
}

fn modelFile(b: *std.Build, model: Model, name: []const u8) std.Build.LazyPath {
    return model.directory.path(b, name);
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
    const dest = if (model.local) b.fmt("local/{s}", .{model.id}) else model.id;
    return b.addInstallArtifact(wasm, .{ .dest_dir = .{ .override = .{ .custom = dest } } });
}
