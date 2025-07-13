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
        const model_path = try getModelPath(allocator);
        const model_name = getModelName(allocator);
        defer allocator.free(model_name);

        const advanced_configs_enabled = isUsingAdvancedConfigs();
        if (advanced_configs_enabled) {

            Util.display(Messages.zllmchat_yellow ++ Messages.ansii_blue ++ "MODEL PARAMS" ++ Messages.ansii_reset ++ "\n");

            const gpu_enabled = isUsingGPU();

            const gpu_layer_count = if(gpu_enabled) getGPULayerCount() else 0;
            const main_gpu_index = if(gpu_enabled) getMainGPUIndex() else 0;
            const tensor_split_mode = if(gpu_enabled) try getTensorSplitMode() else SplitMode.NoSplit;
            const tensor_split_ratios = if(gpu_enabled and tensor_split_mode != SplitMode.NoSplit) getTensorSplitRatios(allocator) else null;
            defer if(tensor_split_ratios) |ratios| allocator.free(ratios);
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
            const pooling_type = try getPoolingType();
            const attention_type = try getAttentionType();
            const rope_scaling_type = try getRopeScalingType();
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
            const key_type = try getKeyType();
            const value_type = if(gpu_enabled) try getValueType() else GGMLType.F16;

            Util.display(Messages.zllmchat_yellow ++ Messages.ansii_blue ++ "SAMPLING CHAIN PARAMS" ++ Messages.ansii_reset ++ "\n");

            var sampling_params = Std.ArrayList(Sampler).init(allocator);

            try addMinPSampling(&sampling_params);
            try addTemperatureSampling(&sampling_params);
            try addDistributionSampling(&sampling_params);
            try addGreedyDecodingSampling(&sampling_params);
            try addTopKSampling(&sampling_params);
            try addTopPSampling(&sampling_params);
            try addTypicalSampling(&sampling_params);
            try addTemperatureAdvancedSampling(&sampling_params);
            try addExtremelyTypicalControlledSampling(&sampling_params);
            try addStandardDeviationSampling(&sampling_params);
            try addMirostat(&sampling_params);
            try addSimplifiedMirostat(&sampling_params);
            try addPenaltiesSampling(&sampling_params);
            try addInfillModeSampling(&sampling_params);
            try addDrySampling(&sampling_params, allocator);
            // TODO: Add all samplers
            //try addLogitBiasSampling(&sampling_params, allocator);
            //try addGrammarSampling(&sampling_params, allocator);
            //try addGrammarLazySampling(&sampling_params, allocator);

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

    pub fn initFromJson(allocator: Allocator, comptime path: []const u8) ParamsError!Params {
        const json = @embedFile(path);

        const parsed = Std.json.parseFromSliceLeaky(
            struct {
                model_path: []const u8,
                model_name: []const u8,
                model_params: ModelParams,
                context_params: ContextParams,
                sampling_params: []struct {
                    @"type": []const u8,
                    params: Std.json.Value,
                },    
            },
            allocator,
            json,
            .{},
        ) catch return ParamsError.InvalidJsonFormat;

        var sampling_params = Std.ArrayList(Sampler).init(allocator);

        const sampler_parsing_hashmap = initSamplerParsingHashmap();

        for (parsed.sampling_params) |param| {
            const handle = sampler_parsing_hashmap.get(param.type) orelse return ParamsError.InvalidJsonFormat;
            const sampler = try handle(allocator, param.params);
            sampling_params.append(sampler) catch return ParamsError.AddToSamplingParamsFailed;
        }

        return Params{
            .model_path = parsed.model_path,
            .model_name = parsed.model_name,
            .model_params = .{
                .gpu_layer_count = parsed.model_params.gpu_layer_count,
                .main_gpu_index = parsed.model_params.main_gpu_index,
                .tensor_split_mode = parsed.model_params.tensor_split_mode,
                .tensor_split_ratios = parsed.model_params.tensor_split_ratios,
                .vocab_only_mode = parsed.model_params.vocab_only_mode,
                .memory_map_enabled = parsed.model_params.memory_map_enabled,
                .memory_lock_enabled = parsed.model_params.memory_lock_enabled,
                .tensor_validation_enabled = parsed.model_params.tensor_validation_enabled,
            },
            .context_params = .{
                .context_size = parsed.context_params.context_size,
                .batch_size = parsed.context_params.batch_size,
                .unified_batch_size = parsed.context_params.unified_batch_size,
                .max_sequence_length = parsed.context_params.max_sequence_length,
                .thread_count = parsed.context_params.thread_count,
                .batch_thread_count = parsed.context_params.batch_thread_count,
                .pooling_type = parsed.context_params.pooling_type,
                .attention_type = parsed.context_params.attention_type,
                .rope_scaling_type = parsed.context_params.rope_scaling_type,
                .rope_frequency_base = parsed.context_params.rope_frequency_base,
                .rope_frequency_scale = parsed.context_params.rope_frequency_scale,
                .yarn_extension_factor = parsed.context_params.yarn_extension_factor,
                .yarn_attention_factor = parsed.context_params.yarn_attention_factor,
                .yarn_beta_fast = parsed.context_params.yarn_beta_fast,
                .yarn_beta_slow = parsed.context_params.yarn_beta_slow,
                .yarn_original_context = parsed.context_params.yarn_original_context,
                .defrag_threshold = parsed.context_params.defrag_threshold,
                .key_type = parsed.context_params.key_type,
                .value_type = parsed.context_params.value_type,
                .all_logits_enabled = parsed.context_params.all_logits_enabled,
                .embeddings_enabled = parsed.context_params.embeddings_enabled,
                .offload_kqv_enabled = parsed.context_params.offload_kqv_enabled,
                .flash_attention_enabled = parsed.context_params.flash_attention_enabled,
                .no_performance_optimizations = parsed.context_params.no_performance_optimizations,
            },
            .sampling_params = sampling_params,
        };
    }
    
    fn initSamplerParsingHashmap() Std.StaticStringMap(*const fn (Allocator, Std.json.Value) ParamsError!Sampler) {

        return Std.StaticStringMap(*const fn (Allocator, Std.json.Value) ParamsError!Sampler).initComptime(
            .{
                .{
                    "MinP", struct {
                        fn handle(allocator: Allocator, params: Std.json.Value) ParamsError!Sampler {
                            const p = params.object.get("p").?.float;
                            const min_keep = @as(usize, @intCast(params.object.get("min_keep").?.integer));
                            const min_p = allocator.create(SamplingTypes.MinP) catch return ParamsError.AllocationFailed;
                            min_p.* = .{
                                .p = @floatCast(p),
                                .min_keep = min_keep
                            };

                            return Sampler.init(SamplingTypes.MinP, .MinP, min_p);
                        }
                    }.handle
                },
                .{
                    "Temperature", struct {
                        fn handle(allocator: Allocator, params: Std.json.Value) ParamsError!Sampler {
                            const temp = params.object.get("temp").?.float;

                            const temperature = allocator.create(SamplingTypes.Temperature) catch return ParamsError.AllocationFailed;

                            temperature.*.temp = @floatCast(temp);

                            return Sampler.init(SamplingTypes.Temperature, .Temperature, temperature);
                        }
                    }.handle
                },
                .{
                    "Distribution", struct {
                        fn handle(allocator: Allocator, params: Std.json.Value) ParamsError!Sampler {
                            const seed = @as(u32, @intCast(params.object.get("seed").?.integer));

                            const distribution = allocator.create(SamplingTypes.Distribution) catch return ParamsError.AllocationFailed;

                            distribution.*.seed = seed;

                            return Sampler.init(SamplingTypes.Distribution, .Distribution, distribution);
                        }
                    }.handle
                },
                .{
                    "GreedyDecoding", struct {
                        fn handle(allocator: Allocator, _: Std.json.Value) ParamsError!Sampler {
                            const greedy_decoding = allocator.create(SamplingTypes.GreedyDecoding) catch return ParamsError.AllocationFailed;
                            
                            return Sampler.init(SamplingTypes.GreedyDecoding, .GreedyDecoding, greedy_decoding);
                        }
                    }.handle
                },
                .{
                    "TopK", struct {
                        fn handle(allocator: Allocator, params: Std.json.Value) ParamsError!Sampler {
                            const k = @as(i32, @intCast(params.object.get("k").?.integer));
                            
                            const top_k = allocator.create(SamplingTypes.TopK) catch return ParamsError.AllocationFailed;

                            top_k.*.k = k;

                            return Sampler.init(SamplingTypes.TopK, .TopK, top_k);
                        }
                    }.handle
                },
                .{
                    "TopP", struct {
                        fn handle(allocator: Allocator, params: Std.json.Value) ParamsError!Sampler {
                            const p = @as(f32, @floatCast(params.object.get("p").?.float));
                            const min_keep = @as(usize, @intCast(params.object.get("min_keep").?.integer));
                        
                            const top_p = allocator.create(SamplingTypes.TopP) catch return ParamsError.AllocationFailed;
                        
                            top_p.* = .{
                                .p = p,
                                .min_keep = min_keep
                            };

                            return Sampler.init(SamplingTypes.TopP, .TopP, top_p);
                        }
                    }.handle
                },
                .{
                    "Typical", struct {
                        fn handle(allocator: Allocator, params: Std.json.Value) ParamsError!Sampler {
                            const p = @as(f32, @floatCast(params.object.get("p").?.float));
                            const min_keep = @as(usize, @intCast(params.object.get("min_keep").?.integer));
                        
                            const typical = allocator.create(SamplingTypes.Typical) catch return ParamsError.AllocationFailed;
                        
                            typical.* = .{
                                .p = p,
                                .min_keep = min_keep
                            };

                            return Sampler.init(SamplingTypes.Typical, .Typical, typical);
                        }
                    }.handle
                },
                .{
                    "TemperatureAdvanced", struct {
                        fn handle(allocator: Allocator, params: Std.json.Value) ParamsError!Sampler {
                            const temp = @as(f32, @floatCast(params.object.get("temp").?.float));
                            const delta = @as(f32, @floatCast(params.object.get("delta").?.float));
                            const exponent = @as(f32, @floatCast(params.object.get("exponent").?.float));

                            const temperature_advanced = allocator.create(SamplingTypes.TemperatureAdvanced) catch return ParamsError.AllocationFailed;

                            temperature_advanced.* = .{
                                .temp = temp,
                                .delta = delta,
                                .exponent = exponent
                            };

                            return Sampler.init(SamplingTypes.TemperatureAdvanced, .TemperatureAdvanced, temperature_advanced);
                        }
                    }.handle
                },
                .{
                    "ExtremelyTypicalControlled", struct {
                        fn handle(allocator: Allocator, params: Std.json.Value) ParamsError!Sampler {
                            const p = @as(f32, @floatCast(params.object.get("p").?.float));
                            const temp = @as(f32, @floatCast(params.object.get("temp").?.float));
                            const min_keep = @as(usize, @intCast(params.object.get("min_keep").?.integer));
                            const seed = @as(u32, @intCast(params.object.get("seed").?.integer));
                            
                            const extremely_typical_controlled = allocator.create(SamplingTypes.ExtremelyTypicalControlled) catch return ParamsError.AllocationFailed;

                            extremely_typical_controlled.* = .{
                                .p = p,
                                .temp = temp,
                                .min_keep = min_keep,
                                .seed = seed
                            };

                            return Sampler.init(SamplingTypes.ExtremelyTypicalControlled, .ExtremelyTypicalControlled, extremely_typical_controlled);
                        }
                    }.handle
                },
                .{
                    "StandardDeviation", struct {
                        fn handle(allocator: Allocator, params: Std.json.Value) ParamsError!Sampler {
                            const width = @as(f32, @floatCast(params.object.get("width").?.float));
                            
                            const standard_deviation = allocator.create(SamplingTypes.StandardDeviation) catch return ParamsError.AllocationFailed;

                            standard_deviation.*.width = width;

                            return Sampler.init(SamplingTypes.StandardDeviation, .StandardDeviation, standard_deviation);
                        }
                    }.handle
                },
                .{
                    "Mirostat", struct {
                        fn handle(allocator: Allocator, params: Std.json.Value) ParamsError!Sampler {
                            const seed = @as(u32, @intCast(params.object.get("seed").?.integer));
                            const window_size = @as(i32, @intCast(params.object.get("window_size").?.integer));
                            const target_surprise = @as(f32, @floatCast(params.object.get("target_surprise").?.float));
                            const learning_rate = @as(f32, @floatCast(params.object.get("learning_rate").?.float));

                            const mirostat = allocator.create(SamplingTypes.Mirostat) catch return ParamsError.AllocationFailed;

                            mirostat.* = .{
                                .seed = seed,
                                .window_size = window_size,
                                .target_surprise = target_surprise,
                                .learning_rate = learning_rate
                            };

                            return Sampler.init(SamplingTypes.Mirostat, .Mirostat, mirostat);
                        }
                    }.handle
                },
                .{
                    "SimplifiedMirostat", struct {
                        fn handle(allocator: Allocator, params: Std.json.Value) ParamsError!Sampler {
                            const seed = @as(u32, @intCast(params.object.get("seed").?.integer));
                            const target_surprise = @as(f32, @floatCast(params.object.get("target_surprise").?.float));
                            const learning_rate = @as(f32, @floatCast(params.object.get("learning_rate").?.float));

                            const simplified_mirosat = allocator.create(SamplingTypes.SimplifiedMirostat) catch return ParamsError.AllocationFailed;

                            simplified_mirosat.* = .{
                                .seed = seed,
                                .target_surprise = target_surprise,
                                .learning_rate = learning_rate
                            };

                            return Sampler.init(SamplingTypes.SimplifiedMirostat, .SimplifiedMirostat, simplified_mirosat);   
                        }
                    }.handle
                },
                .{
                    "Penalties", struct {
                        fn handle(allocator: Allocator, params: Std.json.Value) ParamsError!Sampler {
                            const penalty_last_window = @as(i32, @intCast(params.object.get("penalty_last_window").?.integer));
                            const penalty_repeat = @as(f32, @floatCast(params.object.get("penalty_repeat").?.float));
                            const penalty_frequency = @as(f32, @floatCast(params.object.get("penalty_frequency").?.float));
                            const penality_present = @as(f32, @floatCast(params.object.get("penality_present").?.float));

                            const penalties = allocator.create(SamplingTypes.Penalties) catch return ParamsError.AllocationFailed;

                            penalties.* = .{
                                .penalty_last_window = penalty_last_window,
                                .penalty_repeat = penalty_repeat,
                                .penalty_frequency = penalty_frequency,
                                .penality_present = penality_present
                            };

                            return Sampler.init(SamplingTypes.Penalties, .Penalties, penalties);   
                        }
                    }.handle
                },
                .{
                    "InfillMode", struct {
                        fn handle(allocator: Allocator, _: Std.json.Value) ParamsError!Sampler {
                            const infill_mode = allocator.create(SamplingTypes.InfillMode) catch return ParamsError.AllocationFailed;
                            
                            return Sampler.init(SamplingTypes.InfillMode, .InfillMode, infill_mode);
                        }
                    }.handle
                },
                .{
                    "Dry", struct {
                        fn handle(allocator: Allocator, params: Std.json.Value) ParamsError!Sampler {
                            const train_context_size = @as(i32, @intCast(params.object.get("train_context_size").?.integer));
                            const allowed_length = @as(i32, @intCast(params.object.get("allowed_length").?.integer));
                            const penality_last_window = @as(i32, @intCast(params.object.get("penality_last_window").?.integer));
                            const num_breakers = @as(usize, @intCast(params.object.get("num_breakers").?.integer));
                            const multiplier = @as(f32, @floatCast(params.object.get("multiplier").?.float));
                            const base = @as(f32, @floatCast(params.object.get("base").?.float));
                            
                            const breakers_array = params.object.get("breakers").?.array;

                            var breakers = Std.ArrayList([*c]const u8).init(allocator);
                            for (breakers_array.items) |item| {
                                breakers.append(item.string.ptr) catch return ParamsError.AddToBreakersFailed;
                            }

                            const dry = allocator.create(SamplingTypes.Dry) catch return ParamsError.AllocationFailed;
                            
                            dry.* = .{
                                .train_context_size = train_context_size,
                                .allowed_length = allowed_length,
                                .penality_last_window = penality_last_window,
                                .num_breakers = num_breakers,
                                .multiplier = multiplier,
                                .base = base,
                                .breakers = breakers.items
                            };

                            return Sampler.init(SamplingTypes.Dry, .Dry, dry);
                        }
                    }.handle
                }
            }
        );
    }

    pub fn getDefaultParams(allocator: Allocator, model_path: []const u8, model_name: []const u8, gpu_layer_count: u32, context_size: u32) ParamsError!Params {
        
        const sampling_params = Std.ArrayList(Sampler).init(allocator);

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
            .sampling_params = sampling_params, // Empty
        };
    }

    fn getModelPath(allocator: Allocator) ParamsError![]const u8 {
        const string = consumeString(allocator, Messages.params_model_path, Messages.ansii_underlined, Messages.ansii_clear_1line);
        defer allocator.free(string);
        
        if (!Std.mem.endsWith(u8, string, ".gguf")) {
            return ParamsError.InvalidModelFormat;
        }

        const file = Std.fs.cwd().openFile(string, .{}) catch {
            return ParamsError.FileReadFailed;
        }; 
        defer file.close();

        const model_path = allocator.dupe(u8, string) catch return ParamsError.AllocationFailed;
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

    fn getTensorSplitMode() ParamsError!SplitMode {
        const value = consumeNumber(Messages.params_tensor_split_mode, Messages.ansii_yellow, Messages.ansii_clear_4line, u32, 1);

        const tags = Std.meta.tags(SplitMode);
        for (tags) |tag| {
            if (@intFromEnum(tag) == value) {
                return tag;
            }
        }

        return ParamsError.InvalidValue;
    }

    fn getTensorSplitRatios(allocator: Allocator) []const f32 {
        const str_input = consumeString(allocator, Messages.params_tensor_split_ratios, Messages.ansii_green, Messages.ansii_clear_2line);
        defer allocator.free(str_input);
        var iter = Std.mem.splitAny(u8, str_input, " ");

        var count: usize = 0;
        var tmp_iter = iter;
        while (tmp_iter.next()) |_| {
            count += 1;
        }

        var ratio_array = allocator.alloc(f32, count) catch unreachable;

        iter = Std.mem.splitAny(u8, str_input, " ");
        var i: usize = 0;
        while (iter.next()) |ratio| {
            ratio_array[i] = Std.fmt.parseFloat(f32, ratio) catch unreachable;
            i += 1;
        }

        return ratio_array;
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

    fn getPoolingType() ParamsError!PoolingType {
        const value = consumeNumber(
            Messages.params_pooling_type,
            Messages.ansii_yellow,
            Messages.ansii_clear_8line,
            i32,
            -1
        );

        const tags = Std.meta.tags(PoolingType);
        for (tags) |tag| {
            if (@intFromEnum(tag) == value) {
                return tag;
            }
        }

        return ParamsError.InvalidValue;
    }

    fn getAttentionType() ParamsError!AttentionType {
        const value = consumeNumber(Messages.params_attention_type, Messages.ansii_yellow, Messages.ansii_clear_5line, i32, -1);

        const tags = Std.meta.tags(AttentionType);
        for (tags) |tag| {
            if (@intFromEnum(tag) == value) {
                return tag;
            }
        }

        return ParamsError.InvalidValue;
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

    fn getRopeScalingType() ParamsError!RopeScalingType {
        const value = consumeNumber(Messages.params_rope_scaling_type, Messages.ansii_yellow, Messages.ansii_clear_7line, i32, -1);

        const tags = Std.meta.tags(RopeScalingType);
        for (tags) |tag| {
            if (@intFromEnum(tag) == value) {
                return tag;
            }
        }

        return ParamsError.InvalidValue;
    }

    fn getKeyType() ParamsError!GGMLType {
        const value = consumeNumber(Messages.params_key_type, Messages.ansii_yellow, Messages.ansii_clear_7line, u32, 1);

        const tags = Std.meta.tags(GGMLType);
        for (tags) |tag| {
            if (@intFromEnum(tag) == value) {
                return tag;
            }
        }

        return ParamsError.InvalidValue;
    }

    fn getValueType() ParamsError!GGMLType {
        const value = consumeNumber(Messages.params_value_type, Messages.ansii_yellow, Messages.ansii_clear_6line, u32, 1);

        const tags = Std.meta.tags(GGMLType);
        for (tags) |tag| {
            if (@intFromEnum(tag) == value) {
                return tag;
            }
        }

        return ParamsError.InvalidValue;
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

    fn addMinPSampling(sampling_params: *Std.ArrayList(Sampler)) ParamsError!void {
        const should_use = consumeYesOrNo(Messages.sampling_min_p, Messages.ansii_clear_1line, false);
        if(!should_use) return;

        const p = consumeNumber(Messages.sampling_min_p_p, Messages.ansii_blue, Messages.ansii_clear_1line, f32, 0.05);
        const min_keep = consumeNumber(Messages.sampling_min_p_min_keep, Messages.ansii_purple, Messages.ansii_clear_1line, usize, 1);

        const min_P = SamplingTypes.MinP {
            .p = p,
            .min_keep = min_keep
        };
        const min_p_sampling = Sampler.init(SamplingTypes.MinP, SamplingTypes.TypeName.MinP, &min_P);
        sampling_params.append(min_p_sampling) catch return ParamsError.AddToSamplingParamsFailed;
    }

    fn addTemperatureSampling(sampling_params: *Std.ArrayList(Sampler)) ParamsError!void {
        const should_use = consumeYesOrNo(Messages.sampling_temperature, Messages.ansii_clear_1line, false);
        if(!should_use) return;

        const temp = consumeNumber(Messages.sampling_temperature_temp, Messages.ansii_green, Messages.ansii_clear_1line, f32, 0.08);

        const temperature = SamplingTypes.Temperature {
            .temp = temp
        };
        const temperature_sampling = Sampler.init(SamplingTypes.Temperature, SamplingTypes.TypeName.Temperature, &temperature);
        sampling_params.append(temperature_sampling) catch return ParamsError.AddToSamplingParamsFailed;
    }

    fn addDistributionSampling(sampling_params: *Std.ArrayList(Sampler)) ParamsError!void {
        const should_use = consumeYesOrNo(Messages.sampling_distribution, Messages.ansii_clear_1line, false);
        if(!should_use) return;

        const seed = consumeNumber(Messages.sampling_distribution_seed, Messages.ansii_yellow, Messages.ansii_clear_1line, u32, 0xFFFFFFFF);
        
        const distribution = SamplingTypes.Distribution {
            .seed = seed
        };
        const distribution_sampling = Sampler.init(SamplingTypes.Distribution, SamplingTypes.TypeName.Distribution, &distribution);
        sampling_params.append(distribution_sampling) catch return ParamsError.AddToSamplingParamsFailed;
    }

    fn addGreedyDecodingSampling(sampling_params: *Std.ArrayList(Sampler)) ParamsError!void {
        const should_use = consumeYesOrNo(Messages.sampling_greedy_decoding, Messages.ansii_clear_1line, false);
        if(!should_use) return;
        
        const greedy_decoding = SamplingTypes.GreedyDecoding {};
        const greedy_decoding_sampling = Sampler.init(SamplingTypes.GreedyDecoding, SamplingTypes.TypeName.GreedyDecoding, &greedy_decoding);
        sampling_params.append(greedy_decoding_sampling) catch return ParamsError.AddToSamplingParamsFailed;
    }

    fn addTopKSampling(sampling_params: *Std.ArrayList(Sampler)) ParamsError!void {
        const should_use = consumeYesOrNo(Messages.sampling_top_k, Messages.ansii_clear_1line, false);
        if(!should_use) return;

        const k = consumeNumber(Messages.sampling_top_k_k, Messages.ansii_purple, Messages.ansii_clear_1line, i32, 40);
        
        const top_k = SamplingTypes.TopK {
            .k = k
        };
        const top_k_sampling = Sampler.init(SamplingTypes.TopK, SamplingTypes.TypeName.TopK, &top_k);
        sampling_params.append(top_k_sampling) catch return ParamsError.AddToSamplingParamsFailed;
    }

    fn addTopPSampling(sampling_params: *Std.ArrayList(Sampler)) ParamsError!void {
        const should_use = consumeYesOrNo(Messages.sampling_top_p, Messages.ansii_clear_1line, false);
        if(!should_use) return;

        const p = consumeNumber(Messages.sampling_top_p_p, Messages.ansii_purple, Messages.ansii_clear_1line, f32, 0.85);
        const min_keep = consumeNumber(Messages.sampling_top_p_min_keep, Messages.ansii_green, Messages.ansii_clear_1line, usize, 10);

        const top_p = SamplingTypes.TopP {
            .p = p,
            .min_keep = min_keep
        };
        const top_p_sampling = Sampler.init(SamplingTypes.TopP, SamplingTypes.TypeName.TopP, &top_p);
        sampling_params.append(top_p_sampling) catch return ParamsError.AddToSamplingParamsFailed;
    }

    fn addTypicalSampling(sampling_params: *Std.ArrayList(Sampler)) ParamsError!void {
        const should_use = consumeYesOrNo(Messages.sampling_typical, Messages.ansii_clear_1line, false);
        if(!should_use) return;
        
        const p = consumeNumber(Messages.sampling_typical_p, Messages.ansii_blue, Messages.ansii_clear_1line, f32, 0.95);
        const min_keep = consumeNumber(Messages.sampling_typical_min_keep, Messages.ansii_red, Messages.ansii_clear_1line, usize, 10);

        const typical = SamplingTypes.Typical {
            .p = p,
            .min_keep = min_keep
        };
        const typical_sampling = Sampler.init(SamplingTypes.Typical, SamplingTypes.TypeName.Typical, &typical);
        sampling_params.append(typical_sampling) catch return ParamsError.AddToSamplingParamsFailed;
    }

    fn addTemperatureAdvancedSampling(sampling_params: *Std.ArrayList(Sampler)) ParamsError!void {
        const should_use = consumeYesOrNo(Messages.sampling_temperature_advanced, Messages.ansii_clear_1line, false);
        if(!should_use) return;
        
        const temp = consumeNumber(Messages.sampling_temperature_advanced_temp, Messages.ansii_purple, Messages.ansii_clear_1line, f32, 0.8);
        const delta = consumeNumber(Messages.sampling_temperature_advanced_delta, Messages.ansii_green, Messages.ansii_clear_1line, f32, 0.2);
        const exponent = consumeNumber(Messages.sampling_temperature_advanced_exponent, Messages.ansii_red, Messages.ansii_clear_1line, f32, 1.5);

        const temperature_advanced = SamplingTypes.TemperatureAdvanced {
            .temp = temp,
            .delta = delta,
            .exponent = exponent
        };
        const temperature_advanced_sampling = Sampler.init(SamplingTypes.TemperatureAdvanced, SamplingTypes.TypeName.TemperatureAdvanced, &temperature_advanced);
        sampling_params.append(temperature_advanced_sampling) catch return ParamsError.AddToSamplingParamsFailed;
    }

    fn addExtremelyTypicalControlledSampling(sampling_params: *Std.ArrayList(Sampler)) ParamsError!void {
        const should_use = consumeYesOrNo(Messages.sampling_typical_controlled, Messages.ansii_clear_1line, false);
        if(!should_use) return;

        const p = consumeNumber(Messages.sampling_typical_controlled_p, Messages.ansii_red, Messages.ansii_clear_1line, f32, 0.9);
        const temp = consumeNumber(Messages.sampling_typical_controlled_temp, Messages.ansii_blue, Messages.ansii_clear_1line, f32, 0.7);
        const min_keep = consumeNumber(Messages.sampling_typical_controlled_min_keep, Messages.ansii_red, Messages.ansii_clear_1line, usize, 5);
        const seed = consumeNumber(Messages.sampling_typical_controlled_seed, Messages.ansii_blue, Messages.ansii_clear_1line, u32, 42);

        const extremely_typical_controlled = SamplingTypes.ExtremelyTypicalControlled {
            .p = p,
            .temp = temp,
            .min_keep = min_keep,
            .seed = seed
        };
        const ext_typical_controlled_sampling = Sampler.init(SamplingTypes.ExtremelyTypicalControlled, SamplingTypes.TypeName.ExtremelyTypicalControlled, &extremely_typical_controlled);
        sampling_params.append(ext_typical_controlled_sampling) catch return ParamsError.AddToSamplingParamsFailed;
    }

    fn addStandardDeviationSampling(sampling_params: *Std.ArrayList(Sampler)) ParamsError!void {
        const should_use = consumeYesOrNo(Messages.sampling_standard_deviation, Messages.ansii_clear_1line, false);
        if(!should_use) return;
        
        const width = consumeNumber(Messages.sampling_standard_deviation_width, Messages.ansii_blue, Messages.ansii_clear_1line, f32, 1.5);

        const standard_deviation = SamplingTypes.StandardDeviation {
            .width = width
        };
        const std_deviation_sampling = Sampler.init(SamplingTypes.StandardDeviation, SamplingTypes.TypeName.StandardDeviation, &standard_deviation);
        sampling_params.append(std_deviation_sampling) catch return ParamsError.AddToSamplingParamsFailed;
    }
    
    fn addMirostat(sampling_params: *Std.ArrayList(Sampler)) ParamsError!void {
        const should_use = consumeYesOrNo(Messages.sampling_mirostat, Messages.ansii_clear_1line, false);
        if(!should_use) return;
        
        const seed = consumeNumber(Messages.sampling_mirostat_seed, Messages.ansii_red, Messages.ansii_clear_1line, u32, 42);
        const target_surprise = consumeNumber(Messages.sampling_mirostat_target_surprise, Messages.ansii_blue, Messages.ansii_clear_1line, f32, 4.0);
        const learning_rate = consumeNumber(Messages.sampling_mirostat_learning_rate, Messages.ansii_green, Messages.ansii_clear_1line, f32, 0.2);
        const window_size = consumeNumber(Messages.sampling_mirostat_window_size, Messages.ansii_red, Messages.ansii_clear_1line, i32, 100);

        const mirostat = SamplingTypes.Mirostat {
            .seed = seed,
            .target_surprise = target_surprise,
            .learning_rate = learning_rate,
            .window_size = window_size
        };     
        const mirostat_sampling = Sampler.init(SamplingTypes.Mirostat, SamplingTypes.TypeName.Mirostat, &mirostat);
        sampling_params.append(mirostat_sampling) catch return ParamsError.AddToSamplingParamsFailed;
    }
    
    fn addSimplifiedMirostat(sampling_params: *Std.ArrayList(Sampler)) ParamsError!void {
        const should_use = consumeYesOrNo(Messages.sampling_simplified_mirostat, Messages.ansii_clear_1line, false);
        if(!should_use) return;
        
        const seed = consumeNumber(Messages.sampling_simplified_mirostat_seed, Messages.ansii_red, Messages.ansii_clear_1line, u32, 42);
        const target_surprise = consumeNumber(Messages.sampling_simplified_mirostat_target_surprise, Messages.ansii_blue, Messages.ansii_clear_1line, f32, 4.0);
        const learning_rate = consumeNumber(Messages.sampling_simplified_mirostat_learning_rate, Messages.ansii_green, Messages.ansii_clear_1line, f32, 0.2);

        const simplified_mirostat = SamplingTypes.SimplifiedMirostat {
            .seed = seed,
            .target_surprise = target_surprise,
            .learning_rate = learning_rate
        };
        const sampling_simplified_mirostat = Sampler.init(SamplingTypes.SimplifiedMirostat, SamplingTypes.TypeName.SimplifiedMirostat, &simplified_mirostat);
        sampling_params.append(sampling_simplified_mirostat) catch return ParamsError.AddToSamplingParamsFailed;
    }

    fn addPenaltiesSampling(sampling_params: *Std.ArrayList(Sampler)) ParamsError!void {
        const should_use = consumeYesOrNo(Messages.sampling_penalties, Messages.ansii_clear_1line, false);
        if(!should_use) return;
        
        const penalty_last_window = consumeNumber(Messages.sampling_penalties_last_window, Messages.ansii_red, Messages.ansii_clear_1line, i32, 54);
        const penalty_repeat = consumeNumber(Messages.sampling_penalties_repeat, Messages.ansii_red, Messages.ansii_clear_1line, f32, 1.1);
        const penalty_frequency = consumeNumber(Messages.sampling_penalties_frequency, Messages.ansii_green, Messages.ansii_clear_1line, f32, 0.95);
        const penality_present = consumeNumber(Messages.sampling_penalties_present, Messages.ansii_purple, Messages.ansii_clear_1line, f32, 1.05);

        const penalties = SamplingTypes.Penalties {
            .penalty_last_window = penalty_last_window,
            .penalty_repeat = penalty_repeat,
            .penalty_frequency = penalty_frequency,
            .penality_present = penality_present
        };
        const penalties_sampling = Sampler.init(SamplingTypes.Penalties, SamplingTypes.TypeName.Penalties, &penalties);
        sampling_params.append(penalties_sampling) catch return ParamsError.AddToSamplingParamsFailed;
    }

    fn addInfillModeSampling(sampling_params: *Std.ArrayList(Sampler)) ParamsError!void {
        const should_use = consumeYesOrNo(Messages.sampling_infill_mode, Messages.ansii_clear_1line, false);
        if(!should_use) return;
        
        const infill_mode = SamplingTypes.InfillMode {};
        const infill_mode_sampling = Sampler.init(SamplingTypes.InfillMode, SamplingTypes.TypeName.InfillMode, &infill_mode);
        sampling_params.append(infill_mode_sampling) catch return ParamsError.AddToSamplingParamsFailed;
    }

    fn addDrySampling(sampling_params: *Std.ArrayList(Sampler), allocator: Allocator) ParamsError!void {
        const should_use = consumeYesOrNo(Messages.sampling_dry, Messages.ansii_clear_1line, false);
        if (!should_use) return;

        const train_context_size = consumeNumber(Messages.sampling_dry_train_context_size, Messages.ansii_purple, Messages.ansii_clear_1line, i32, 2048);
        const multiplier = consumeNumber(Messages.sampling_dry_multiplier, Messages.ansii_green, Messages.ansii_clear_1line, f32, 1.32);
        const base = consumeNumber(Messages.sampling_dry_base, Messages.ansii_red, Messages.ansii_clear_1line, f32, 0.1);
        const allowed_length = consumeNumber(Messages.sampling_dry_allowed_length, Messages.ansii_blue, Messages.ansii_clear_1line, i32, 50);
        const penality_last_window = consumeNumber(Messages.sampling_dry_penality_last_window, Messages.ansii_yellow, Messages.ansii_clear_1line, i32, 64);
        const breakers_str = consumeString(allocator, Messages.sampling_dry_breakers, Messages.ansii_purple, Messages.ansii_clear_1line);
        defer allocator.free(breakers_str);

        var breakers_list = Std.ArrayList([]const u8).init(allocator);
        defer {
            for (breakers_list.items) |item| {
                allocator.free(item);
            }
            breakers_list.deinit();
        }

        var breakers_iter = Std.mem.splitAny(u8, breakers_str, " ");
        while (breakers_iter.next()) |token| {
            const duped_token = allocator.dupe(u8, token) catch return ParamsError.DupeTokenFailed;
            breakers_list.append(duped_token) catch return ParamsError.AddToBreakersFailed;
        }

        const c_breakers = allocator.alloc([*c]const u8, breakers_list.items.len) catch return ParamsError.AllocationFailed;
        defer allocator.free(c_breakers);

        for (breakers_list.items, 0..) |token, i| {
            c_breakers[i] = token.ptr;
        }

        const dry = SamplingTypes.Dry {
            .train_context_size = train_context_size,
            .multiplier = multiplier,
            .base = base,
            .allowed_length = allowed_length,
            .penality_last_window = penality_last_window,
            .breakers = c_breakers,
            .num_breakers = breakers_list.items.len,
        };

        const dry_sampling = Sampler.init(SamplingTypes.Dry, SamplingTypes.TypeName.Dry, &dry);
        sampling_params.append(dry_sampling) catch return ParamsError.AddToSamplingParamsFailed;
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
};

pub const ModelParams = struct {
    gpu_layer_count: u32,
    main_gpu_index: u32,
    tensor_split_mode: SplitMode,
    tensor_split_ratios: ?[] const f32,
    memory_map_enabled: bool,
    memory_lock_enabled: bool,
    vocab_only_mode: bool,
    tensor_validation_enabled: bool,
    // TODO: Add all params
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
    // TODO: Add all params
    //abort_callback: ?*const fn (?*anyopaque) bool,
    //abort_callback_data: ?*const fn (?*anyopaque) bool,
    //cb_eval: ?*const fn ([*c]struct_ggml_tensor, bool, ?*anyopaque) bool,
    // cb_eval_user_data: ?*anyopaque,
};

pub const Sampler = struct {
    param_type: SamplingTypes.TypeName,
    params: *const anyopaque,

    pub fn init(comptime T: type, type_name: SamplingTypes.TypeName, params: *const T) Sampler {
        return .{
            .param_type = type_name,
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
    DupeTokenFailed,
    AddToBreakersFailed,
    AllocationFailed,
    InvalidValue,
    InvalidJsonPath,
    InvalidJsonFormat,
};
