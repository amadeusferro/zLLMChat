const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const params_from_json = b.option(
        bool, 
        "PARAMS_FROM_JSON",
        "Set to True to config/load parameters from params.json"
    ) orelse false;

    const opts = b.addOptions();
    opts.addOption(bool, "params_from_json", params_from_json);

    const exe = b.addExecutable(.{
        .name = "zLLMChat",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    exe.root_module.addImport("build_options", opts.createModule());

    const llama_sources = &.{
        "llama.cpp/src/llama-adapter.cpp",
        "llama.cpp/src/llama-arch.cpp",
        "llama.cpp/src/llama-batch.cpp",
        "llama.cpp/src/llama-chat.cpp",
        "llama.cpp/src/llama-context.cpp",
        "llama.cpp/src/llama-cparams.cpp",
        "llama.cpp/src/llama.cpp",
        "llama.cpp/src/llama-grammar.cpp",
        "llama.cpp/src/llama-graph.cpp",
        "llama.cpp/src/llama-hparams.cpp",
        "llama.cpp/src/llama-impl.cpp",
        "llama.cpp/src/llama-io.cpp",
        "llama.cpp/src/llama-kv-cache.cpp",
        "llama.cpp/src/llama-memory.cpp",
        "llama.cpp/src/llama-mmap.cpp",
        "llama.cpp/src/llama-model.cpp",
        "llama.cpp/src/llama-model-loader.cpp",
        "llama.cpp/src/llama-quant.cpp",
        "llama.cpp/src/llama-sampling.cpp",
        "llama.cpp/src/llama-vocab.cpp",
        "llama.cpp/src/unicode.cpp",
        "llama.cpp/src/unicode-data.cpp"
    };

    exe.addCSourceFiles(.{
        .files = llama_sources,
        .flags = &.{
            "-std=c++17", 
            "-Illama.cpp/include",
            "-Illama.cpp/ggml/include",
            "-DGGML_USE_CUBLAS",
            "-O3",
            "-pthread",
        },
    });

    exe.addIncludePath(.{ .cwd_relative = "llama.cpp/include" });
    exe.addIncludePath(.{ .cwd_relative = "llama.cpp/ggml/include" });
    exe.addIncludePath(.{ .cwd_relative = "llama.cpp" });

    exe.linkLibC();
    exe.linkLibCpp();
    exe.linkSystemLibrary("pthread");
    exe.linkSystemLibrary("dl");
    
    exe.addLibraryPath(.{ .cwd_relative = "llama.cpp/build/bin" });
    exe.linkSystemLibrary("llama");
    exe.linkSystemLibrary("ggml");
    exe.linkSystemLibrary("ggml-base");
    exe.linkSystemLibrary("ggml-cpu");
    
    exe.addRPath(.{ .cwd_relative = "llama.cpp/build/bin" });

    b.installArtifact(exe);
}