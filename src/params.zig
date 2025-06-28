const Std = @import("std");
const Allocator = Std.mem.Allocator;

const Util = @import("util.zig").Util;
const Messages = @import("messages.zig").Messages;

const SamplingTypes = @import("sampling_types.zig");

pub const Params = struct {
    model_path: []const u8,
    model_name: []const u8,

    model_params: ModelParams,
    context_params: ContextParams,
    sampling_params: Std.ArrayList(Sampler),

    pub fn init(allocator: Allocator) ParamsError!Params {
        const model_path = getModelPath(allocator) catch |err| return err;
        const model_name = getModelName(allocator);

        const advanced_configs_enabled = isUsingAdvancedConfigs();
        if (advanced_configs_enabled) {

            Util.display(Messages.zllmchat_yellow ++ Messages.ansii_blue ++ "MODEL PARAMS" ++ Messages.ansii_reset ++ "\n");

            const gpu_enabled = isUsingGPU();

            const gpu_layer_count = if(gpu_enabled) getGPULayerCount() else 0;
            const main_gpu_index = if(gpu_enabled) getMainGPUIndex() else 0;
            const tensor_split_mode = if(gpu_enabled) getTensorSplitMode() else SplitMode.NoSplit;
            const tensor_split_ratios = if(gpu_enabled and tensor_split_mode != SplitMode.NoSplit) getTensorSplitRatios() else null;
            const vocab_only_mode = isUsingOnlyVocabMode();
            const memory_map_enabled = isUsingMemoryMap();
            const memory_lock_enabled = isUsingMemoryLock();
            const tensor_validation_enabled = isUsingTensorValidation();

            Util.display(Messages.zllmchat_yellow ++ Messages.ansii_blue ++ "CONTEXT PARAMS" ++ Messages.ansii_reset ++ "\n");

            const context_size = getContextSize();
            const batch_size = getBatchSize();
            const unified_batch_size = getUnifiedBatchSize();
            const max_sequence_length = getMaxSequenceLength();
            const thread_count = getThreadCount();
            const batch_thread_count = getBatchThreadCount();
            const pooling_type = getPoolingType();
            const attention_type = getAttentionType();
            const rope_scaling_type = getRopeScalingType();
            const rope_frequency_base = getRopeFrequencyBase();
            const rope_frequency_scale = getRopeFrequencyScale();
            const yarn_extension_factor = if(rope_scaling_type == RopeScalingType.YaRN) getYarnExtensionFactor() else -1.0;
            const yarn_attention_factor = if(rope_scaling_type == RopeScalingType.YaRN) getYarnAttentionFactor() else 1.0;
            const yarn_beta_fast = if(rope_scaling_type == RopeScalingType.YaRN) getYarnBetaFast() else 32.0;
            const yarn_beta_slow = if(rope_scaling_type == RopeScalingType.YaRN) getYarnBetaSlow() else 1.0;
            const yarn_original_context = if(rope_scaling_type == RopeScalingType.YaRN) getYarnOriginalContext() else 0;
            const defrag_threshold = getDefragThreshold();
            const all_logits_enabled = isUsingAllLogits();
            const embeddings_enabled = isUsingEmbeddings();
            const offload_kqv_enabled = if(gpu_enabled) isUsingKQVOffload() else false;
            const flash_attention_enabled = if(gpu_enabled) isUsingFlashAttention() else false; //NVIDIA GPU
            const no_performance_optimizations =  isUsingNoPerfOptimizations();
            const key_type = getKeyType();
            const value_type = if(gpu_enabled) getValueType() else GGMLType.F16;

            Util.display(Messages.zllmchat_yellow ++ Messages.ansii_blue ++ "SAMPLING CHAIN PARAMS" ++ Messages.ansii_reset ++ "\n");

            const sampling_params =Std.ArrayList(Sampler).init(allocator);

            const min_p_sampling = getMinPSampling();
            const temperature_sampling = getTemperatureSampling();
            const distribution_sampling = getDistributionSampling();

            sampling_params.addOne(min_p_sampling) catch return ParamsError.AddToSamplingParamsFailed;
            sampling_params.addOne(temperature_sampling) catch return ParamsError.AddToSamplingParamsFailed;
            sampling_params.addOne(distribution_sampling) catch return ParamsError.AddToSamplingParamsFailed;

            

            

            // TODO: fz pegar values e textinho bonitinho dos samplers
            // TODO: fz uma msg do AddToSamplingParamsFailed e colocar no handler
            // TODO: fazer api de samplers
            // TODO: arrumar SplitRatios (float)
            // TODO: tratar range do SplitMode  PoolingType  AttentionType  RopeScalingType  GGMLType
            // TODO: gravar videos de uso / fazer bash / fazer readme

            return Params{
                .model_path                       = model_path,
                .model_name                       = model_name,
                .model_params                     = ModelParams {
                    .gpu_layer_count              = gpu_layer_count,
                    .main_gpu_index               = main_gpu_index,
                    .tensor_split_mode            = tensor_split_mode,
                    .tensor_split_ratios          = tensor_split_ratios,
                    .vocab_only_mode              = vocab_only_mode,
                    .memory_map_enabled           = memory_map_enabled,
                    .memory_lock_enabled          = memory_lock_enabled,
                    .tensor_validation_enabled    = tensor_validation_enabled,
                },
                .context_params = ContextParams {
                    .context_size                 = context_size,
                    .batch_size                   = batch_size,
                    .unified_batch_size           = unified_batch_size,
                    .max_sequence_length          = max_sequence_length,
                    .thread_count                 = thread_count,
                    .batch_thread_count           = batch_thread_count,
                    .rope_scaling_type            = rope_scaling_type,
                    .pooling_type                 = pooling_type,
                    .attention_type               = attention_type,
                    .rope_frequency_base          = rope_frequency_base,
                    .rope_frequency_scale         = rope_frequency_scale,
                    .yarn_extension_factor        = yarn_extension_factor,
                    .yarn_attention_factor        = yarn_attention_factor,
                    .yarn_beta_fast               = yarn_beta_fast,
                    .yarn_beta_slow               = yarn_beta_slow,
                    .yarn_original_context        = yarn_original_context,
                    .defrag_threshold             = defrag_threshold,
                    .key_type                     = key_type,
                    .value_type                   = value_type,
                    .all_logits_enabled           = all_logits_enabled,
                    .embeddings_enabled           = embeddings_enabled,
                    .offload_kqv_enabled          = offload_kqv_enabled,
                    .flash_attention_enabled      = flash_attention_enabled,
                    .no_performance_optimizations = no_performance_optimizations,
                },
                .sampling_params = sampling_params,
            };
        }

        const context_size = getContextSize();
        const gpu_enabled = isUsingGPU();
        const gpu_layer_count = if(gpu_enabled) getGPULayerCount() else 0;

        return getDefaultParams(allocator, model_path, model_name, gpu_layer_count, context_size);
    }

    pub fn getDefaultParams(allocator: Allocator, model_path: []const u8, model_name: []const u8, gpu_layer_count: u32, context_size: u32) ParamsError!Params {
        
        const sampling_params = Std.ArrayList(Sampler).init(allocator);

        const min_P = SamplingTypes.MinP {
            .p = 0.05,
            .min_keep = 1
        };
        const min_P_sampling = Sampler.init(SamplingTypes.MinP, &min_P);
        sampling_params.addOne(min_P_sampling) catch return ParamsError.AddToSamplingParamsFailed;

        const temperature = SamplingTypes.Temperature {
            .temp = 0.8
        };
        const temperature_sampling = Sampler.init(SamplingTypes.Temperature, &temperature);
        sampling_params.addOne(temperature_sampling) catch return ParamsError.AddToSamplingParamsFailed;

        const distribution = SamplingTypes.Distribution{
            .seed = 0xFFFFFFFF
        };
        const distribution_sampling = Sampler.init(SamplingTypes.Distribution, &distribution);
        sampling_params.addOne(distribution_sampling) catch return ParamsError.AddToSamplingParamsFailed;

        return Params{
            .model_path = model_path,
            .model_name = model_name,
            .model_params = ModelParams {
                .gpu_layer_count                 = gpu_layer_count,
                // default
                .tensor_split_mode               = SplitMode.LayerSplit,
                .main_gpu_index                  = 0,
                .tensor_split_ratios             = null,
                .vocab_only_mode                 = false,
                .memory_map_enabled              = true,
                .memory_lock_enabled             =  false,
                .tensor_validation_enabled       = false,
            },
            .context_params = ContextParams {
                .context_size                    = context_size,
                .batch_size                      = context_size,
                .unified_batch_size              = 512,
                .max_sequence_length             = 1,
                .thread_count                    = 4,
                .batch_thread_count              = 4,
                .pooling_type                    = PoolingType.Unespecified,
                .rope_scaling_type               = RopeScalingType.Unspecified,
                .rope_frequency_base             = 0.0,
                .rope_frequency_scale            = 0.0,
                .yarn_extension_factor           = -1.0,
                .yarn_attention_factor           = 1.0,
                .yarn_beta_fast                  = 32.0,
                .yarn_beta_slow                  = 1.0,
                .yarn_original_context           = 0,
                .defrag_threshold                = -1.0,
                .attention_type                  = AttentionType.Unespecified,
                .key_type                        = GGMLType.F16,
                .value_type                      = GGMLType.F16,
                .all_logits_enabled              = false,
                .embeddings_enabled              = false,
                .offload_kqv_enabled             = true,
                .flash_attention_enabled         = false,
                .no_performance_optimizations    = true,
            },
            .sampling_params = sampling_params,
        };
    }

    fn getModelPath(allocator: Allocator) ParamsError![]const u8 {
        const string = consumeString(allocator, Messages.params_model_path, Messages.ansii_underlined, Messages.ansii_clear_1line);
        const model_path = string;

        if (!Std.mem.endsWith(u8, model_path, ".gguf")) {
            allocator.free(model_path);
            return ParamsError.InvalidModelFormat;
        }

        const file = Std.fs.cwd().openFile(model_path, .{}) catch {
            allocator.free(model_path);
            return ParamsError.FileReadFailed;
        }; 
        defer file.close();

        return model_path;
    }

    fn getModelName(allocator: Allocator) []const u8 {
        return consumeString(allocator, Messages.params_model_name, Messages.ansii_purple, Messages.ansii_clear_1line);
    }

    fn getContextSize() u32 {
        return consumeNumber(Messages.params_context_size, Messages.ansii_yellow, Messages.ansii_clear_1line, u32, 2048);
    }

    fn isUsingGPU() bool {
        return consumeYesOrNo(Messages.params_enable_GPU, Messages.ansii_clear_1line, true);     
    }

    fn getMainGPUIndex() u32 {
        return consumeNumber(Messages.params_main_GPU_index, Messages.ansii_purple, Messages.ansii_clear_1line, u32, 0);
    }

    fn getTensorSplitMode() SplitMode {
        return @enumFromInt(consumeNumber(Messages.params_tensor_split_mode, Messages.ansii_yellow, Messages.ansii_clear_4line, u32, 1));
    }

    fn getTensorSplitRatios() [*c]const f32 {
        return null;
    }

    fn isUsingOnlyVocabMode() bool {
        return consumeYesOrNo(Messages.params_vocab_only_mode, Messages.ansii_clear_1line, false);
    }

    fn isUsingMemoryMap() bool {
        return consumeYesOrNo(Messages.params_memory_map_enabled, Messages.ansii_clear_1line, true);
    }

    fn isUsingMemoryLock() bool {
        return consumeYesOrNo(Messages.params_memory_lock_enabled, Messages.ansii_clear_1line, false);
    }

    fn isUsingTensorValidation() bool {
        return consumeYesOrNo(Messages.tensor_validation_enabled, Messages.ansii_clear_1line, false);
    }

    fn getBatchSize() u32 {
        return consumeNumber(Messages.params_batch_size, Messages.ansii_yellow, Messages.ansii_clear_1line, u32, 2048);
    }

    fn getUnifiedBatchSize() u32 {
        return consumeNumber(Messages.params_unified_batch_size, Messages.ansii_yellow, Messages.ansii_clear_1line, u32, 512);
    }

    fn getMaxSequenceLength() u32 {
        return consumeNumber(Messages.params_max_sequence_length, Messages.ansii_yellow, Messages.ansii_clear_1line, u32, 1);
    }

    fn getThreadCount() u32 {
        return consumeNumber(Messages.params_thread_count, Messages.ansii_yellow, Messages.ansii_clear_1line, u32, 4);
    }
    
    fn getBatchThreadCount() u32 {
        return consumeNumber(Messages.params_batch_thread_count, Messages.ansii_yellow, Messages.ansii_clear_1line, u32, 4);
    }

    fn getPoolingType() PoolingType {
        return @enumFromInt(consumeNumber(Messages.params_pooling_type, Messages.ansii_yellow, Messages.ansii_clear_8line, i32, -1));
    }

    fn getAttentionType() AttentionType {
        return @enumFromInt(consumeNumber(Messages.params_attention_type, Messages.ansii_yellow, Messages.ansii_clear_5line, i32, -1));
    }

    fn getRopeFrequencyBase() f32 {
        return consumeNumber(Messages.params_rope_frequency_base, Messages.ansii_yellow, Messages.ansii_clear_1line, f32, 0.0);    
    }

    fn getRopeFrequencyScale() f32 {
        return consumeNumber(Messages.params_rope_frequency_scale, Messages.ansii_yellow, Messages.ansii_clear_1line, f32, 0.0);    
    }

    fn isUsingAllLogits() bool {
        return consumeYesOrNo(Messages.params_all_logits_enabled, Messages.ansii_clear_1line, false);    
    }

    fn isUsingEmbeddings() bool {
        return consumeYesOrNo(Messages.params_embeddings_enabled, Messages.ansii_clear_1line, false);    
    }

    fn isUsingKQVOffload() bool {
        return consumeYesOrNo(Messages.params_offload_kqv_enabled, Messages.ansii_clear_1line, true);    
    }

    fn isUsingFlashAttention() bool {
        return consumeYesOrNo(Messages.params_flash_attention_enabled, Messages.ansii_clear_1line, false);    
    }

    fn isUsingNoPerfOptimizations() bool {
        return consumeYesOrNo(Messages.params_no_performance_optimizations, Messages.ansii_clear_1line, true);    
    }

    fn getRopeScalingType() RopeScalingType { 
        return @enumFromInt(consumeNumber(Messages.params_rope_scaling_type, Messages.ansii_yellow, Messages.ansii_clear_7line, i32, -1));
    }

    fn getKeyType() GGMLType {
        return @enumFromInt(consumeNumber(Messages.params_key_type, Messages.ansii_yellow, Messages.ansii_clear_7line, u32, 1));
    }

    fn getValueType() GGMLType {
        return @enumFromInt(consumeNumber(Messages.params_value_type, Messages.ansii_yellow, Messages.ansii_clear_7line, u32, 1));
    }

    fn getDefragThreshold() f32 {
        return consumeNumber(Messages.params_defrag_threshold, Messages.ansii_yellow, Messages.ansii_clear_1line, f32, -1.0);    
    }

    fn getYarnExtensionFactor() f32 {
        return consumeNumber(Messages.params_yarn_extension_factor, Messages.ansii_yellow, Messages.ansii_clear_1line, f32, -1.0);
    }

    fn getYarnAttentionFactor() f32 {
        return consumeNumber(Messages.params_yarn_attention_factor, Messages.ansii_yellow, Messages.ansii_clear_1line, f32, 1.0);
    }

    fn getYarnBetaFast() f32 {
        return consumeNumber(Messages.params_yarn_beta_fast, Messages.ansii_yellow, Messages.ansii_clear_1line, f32, 32.0);
    }

    fn getYarnBetaSlow() f32 {
        return consumeNumber(Messages.params_yarn_beta_slow, Messages.ansii_yellow, Messages.ansii_clear_1line, f32, 1.0);
    }

    fn getYarnOriginalContext() u32 {
        return consumeNumber(Messages.params_yarn_original_context, Messages.ansii_yellow, Messages.ansii_clear_1line, u32, 0);    
    }

    fn getGPULayerCount() u32 {
        return consumeNumber(Messages.params_GPU_layer_count, Messages.ansii_yellow, Messages.ansii_clear_1line, u32, 999);
    }

    fn isUsingAdvancedConfigs() bool {
        return consumeYesOrNo(Messages.params_advanced_configs, Messages.ansii_clear_1line, false);
    }

    fn consumeString(allocator: Allocator, comptime message: []const u8, comptime adornment: []const u8, comptime ansii_clear_line: []const u8) []const u8 {
        var string: ?[]const u8 = null;
        Util.display(
            Messages.zllmchat_yellow ++
            Messages.question_mark ++
            message);
        string = Util.readString(allocator);
        while(string == null) {
            Util.display(
                ansii_clear_line ++
                Messages.zllmchat_yellow ++
                Messages.question_mark ++
                message);
            string = Util.readString(allocator);
        }
        Util.display(
            ansii_clear_line ++
            Messages.zllmchat_yellow ++
            Messages.check_mark ++ 
            message ++
            adornment);
        Util.display(string.?);
        Util.display(Messages.ansii_reset ++ Messages.breakrow);
        return string.?;
    }

    fn consumeNumber(comptime message: []const u8, comptime adornment: []const u8, comptime ansii_clear_line: []const u8, comptime T: type, default: T) T {
        var number: ?T = null;
        Util.display(
            Messages.zllmchat_yellow ++
            Messages.question_mark ++
            message);
        number = Util.readNumber(T, default);
        while(number == null) {
            Util.display(
                ansii_clear_line ++
                Messages.zllmchat_yellow ++
                Messages.question_mark ++
                message);
            number = Util.readNumber(T, default);
        }
        Util.display(
            ansii_clear_line ++
            Messages.zllmchat_yellow ++
            Messages.check_mark ++
            message ++
            adornment);
        Util.displayNumber(T, number.?);
        Util.display(Messages.ansii_reset ++ Messages.breakrow);
        return number.?;
    }

    fn consumeYesOrNo(comptime message: []const u8, comptime ansii_clear_line: []const u8, default: bool) bool {
        var question: ?bool = null;
        Util.display(
            Messages.zllmchat_yellow ++
            Messages.question_mark ++
            message);
        question = Util.readBool(default);
        while(question == null) {
            Util.display(
                ansii_clear_line ++
                Messages.zllmchat_yellow ++
                Messages.question_mark ++
                message);
            question = Util.readBool(default);
        }
        Util.display(
            ansii_clear_line ++
            Messages.zllmchat_yellow ++
            Messages.check_mark ++ 
            message);
        if(question.?) Util.display(Messages.ansii_green ++ "Yes" ++ Messages.ansii_reset ++ "\n") else Util.display(Messages.ansii_red ++ "Not" ++ Messages.ansii_reset ++ "\n");
        return question.?;
    }

    fn isUsingDefaultSampling() bool {

    }

    fn getMinPSampling() Sampler {
        const min_P = SamplingTypes.MinP {
            .p = 0.05,
            .min_keep = 1
        };
        return Sampler.init(SamplingTypes.MinP, &min_P);
    }

    fn getTemperatureSampling() Sampler {
        const temperature = SamplingTypes.Temperature {
            .temp = 0.8
        };
        return Sampler.init(SamplingTypes.Temperature, &temperature);
    }

    fn getDistributionSampling() Sampler {
        const distribution = SamplingTypes.Distribution {
            .seed = 0xFFFFFFFF
        };
        return Sampler.init(SamplingTypes.Distribution, &distribution);
    }
};

pub const ModelParams = struct {
    gpu_layer_count: u32,
    main_gpu_index: u32,
    tensor_split_mode: SplitMode,
    tensor_split_ratios: [*c] const f32,
    memory_map_enabled: bool,
    memory_lock_enabled: bool,
    vocab_only_mode: bool,
    tensor_validation_enabled: bool,
    // TODO
    // devices: [*c]?*struct_ggml_backend_device,
    // tensor_buft_overrides: [*c]*const struct_llama_model_tensor_buft_override,
    // progress_callback: ?*const fn (f32, ?*anyopaque) bool,
    // progress_callback_user_data: ?*anyopaque,
    // kv_overrides: [*c]const struct_llama_model_kv_override,
};

pub const ContextParams = struct {
    context_size: u32,
    batch_size: u32,
    unified_batch_size: u32,
    max_sequence_length: u32,
    thread_count: u32,
    batch_thread_count: u32,
    pooling_type: PoolingType,
    attention_type: AttentionType,
    rope_scaling_type: RopeScalingType,
    rope_frequency_base: f32,
    rope_frequency_scale: f32,
    yarn_extension_factor: f32,
    yarn_attention_factor: f32,
    yarn_beta_fast: f32,
    yarn_beta_slow: f32,
    yarn_original_context: u32,
    defrag_threshold: f32,
    key_type: GGMLType,
    value_type: GGMLType,
    all_logits_enabled: bool,
    embeddings_enabled: bool,
    offload_kqv_enabled: bool,
    flash_attention_enabled: bool,
    no_performance_optimizations: bool,
    // TODO
    //abort_callback: ?*const fn (?*anyopaque) bool,
    //abort_callback_data: ?*const fn (?*anyopaque) bool,
    //cb_eval: ?*const fn ([*c]struct_ggml_tensor, bool, ?*anyopaque) bool,
    // cb_eval_user_data: ?*anyopaque,
};

pub const Sampler = struct {
    param_type: type,
    params: *const anyopaque,

    pub fn init(comptime T: type, params: *const T) Sampler {
        return .{
            .param_type = T,
            .params = @ptrCast(params)
        };
    }

    pub fn getParams(self: Sampler, comptime T: type) *const T {
        return @ptrCast(@alignCast(self.params));
    }
};

const SplitMode = enum(u8) {
    NoSplit = 0,
    LayerSplit = 1,
    RowSplit = 2,
};

const PoolingType = enum(i8) {
    Unespecified = -1,
    None = 0,
    Mean = 1,
    CLS = 2,
    Last = 3,
    Rank = 4,
};

const AttentionType = enum(i8) {
    Unespecified = -1,
    MaskedSelfAttention = 0,
    FullSelfAttention = 1,
};

const RopeScalingType = enum(i8) {
    Unspecified = -1,
    None = 0,
    Linear = 1,
    YaRN = 2,
    LongRoPe = 3,
    MaxValue = 4,
};       

const GGMLType = enum(u8) {
    F32 = 0,
    F16 = 1,
    BF16 = 30,
    Q4_K = 12,
    Q8_0 = 8,
};   

pub const ParamsError = error {
    InvalidModelFormat,
    FileReadFailed,
    AddToSamplingParamsFailed,
};
