pub const Messages = struct {
    // ANSI utilities
    pub const ansii_clear_1line: []const u8 = "\x1b[1A\r\x1b[2K";
    pub const ansii_clear_2line: []const u8 = "\x1b[2A\r\x1b[2K";
    pub const ansii_clear_3line: []const u8 = "\x1b[3A\r\x1b[2K";
    pub const ansii_clear_4line: []const u8 = "\x1b[4A\r\x1b[2K";
    pub const ansii_clear_5line: []const u8 = "\x1b[5A\r\x1b[2K";
    pub const ansii_clear_6line: []const u8 = "\x1b[6A\r\x1b[2K";
    pub const ansii_clear_7line: []const u8 = "\x1b[7A\r\x1b[2K";
    pub const ansii_clear_8line: []const u8 = "\x1b[8A\r\x1b[2K";
    pub const ansii_purple: []const u8 = "\x1b[38;2;200;100;255m";
    pub const ansii_yellow: []const u8 = "\x1b[33m";
    pub const ansii_red: []const u8 = "\x1b[31m";
    pub const ansii_green: []const u8 = "\x1b[32m";
    pub const ansii_blue: []const u8 = "\x1b[38;5;117m";
    pub const ansii_underlined: []const u8 = "\x1b[4m";
    pub const ansii_clear_console: []const u8 = "\x1B[2J\x1B[H" ++ "\x1Bc" ++ "\x1B[3J";
    pub const ansii_reset: []const u8 = "\x1b[0m";
    // General utilities
    pub const breakrow: []const u8 = "\n";
    pub const greaterthan: []const u8 = "> ";
    pub const greaterthan_purple: []const u8 = "\x1b[38;2;200;100;255m>\x1b[0m ";
    pub const question_mark: []const u8 = "\x1b[1;31m?\x1b[0m ";
    pub const check_mark: []const u8 = "\x1b[1;32m✔\x1b[0m ";
    pub const exclamation_mark: []const u8 = "\x1b[1;31m!\x1b[0m ";
    pub const cursor: []const u8 = "\n=> ";
    pub const cursor_yellow: []const u8 = "\n\x1b[33m=>\x1b[0m ";
    pub const zllmchat: []const u8 = "zLLMChat> ";
    pub const zllmchat_yellow: []const u8 = "\x1b[33mzLLMChat>\x1b[0m ";
    pub const util_green: []const u8 = "\x1b[38;2;102;255;204mUtil>\x1b[0m ";
    // Params
    pub const params_model_path: []const u8 = "Model file path \x1b[38;5;245m(.gguf location)\x1b[0m: ";
    pub const params_model_name: []const u8 = "Model name \x1b[38;5;245m(arbitrary name)\x1b[0m: ";
    pub const params_advanced_configs: []const u8 = "Advanced configs \x1b[38;5;245m(y/n)\x1b[0m: ";
    // Model Params
    pub const params_enable_GPU: []const u8 = "Enable GPU \x1b[38;5;245m(default:y) (y/n)\x1b[0m: ";
    pub const params_GPU_layer_count: []const u8 = "GPU layer count \x1b[38;5;245m(default:999)\x1b[0m: ";
    pub const params_main_GPU_index: []const u8 = "Main GPU index \x1b[38;5;245m(used in multi-GPU systems; default:0)\x1b[0m: ";
    pub const params_tensor_split_mode: []const u8 = "Tensor split mode \x1b[38;5;245m(default:1;\n                               0 = No split (single-GPU);\n                               1 = Layer split (each GPU processes different layers);\n                               2 = Row split (horizontal tensor split))\x1b[0m: ";
    pub const params_tensor_split_ratios: []const u8 = "Tensor split ratios \x1b[38;5;245m(Defines the memory split ratio between GPUs;\n                                 Example: 0.7 0.3 (Allocates 70% in GPU0 and 30% in GPU1))\x1b[0m: ";
    pub const params_vocab_only_mode: []const u8 = "Enable vocab only mode \x1b[38;5;245m(Loads only the model's vocabulary; default:n) (y/n)\x1b[0m: ";
    pub const params_memory_map_enabled: []const u8 = "Enable memory map \x1b[38;5;245m(default:y) (y/n)\x1b[0m: ";
    pub const params_memory_lock_enabled: []const u8 = "Enable memory lock \x1b[38;5;245m(avoids swap; default:n) (y/n)\x1b[0m: ";
    pub const tensor_validation_enabled: []const u8 = "Enable tensor validation \x1b[38;5;245m(default:n) (y/n)\x1b[0m: ";
    // Context Params
    pub const params_context_size: []const u8 = "Context size \x1b[38;5;245m(tokens memory, default:2048)\x1b[0m: ";
    pub const params_batch_size: []const u8 = "Batch size \x1b[38;5;245m(tokens processed simultaneously during inference; default:2048)\x1b[0m: ";
    pub const params_unified_batch_size: []const u8 = "Unifed batch size \x1b[38;5;245m(Optimizes processing with prompts of different sizes; default:512)\x1b[0m: ";
    pub const params_max_sequence_length: []const u8 = "Max sequence length \x1b[38;5;245m(Upper limit for continuous text generation; default:1)\x1b[0m: ";
    pub const params_thread_count: []const u8 = "Inference threads \x1b[38;5;245m(default:4)\x1b[0m: ";
    pub const params_batch_thread_count: []const u8 = "Batch threads \x1b[38;5;245m(default:4)\x1b[0m: ";
    pub const params_pooling_type: []const u8 = "Pooling type \x1b[38;5;245m(how embeddings are aggregated in sequence outputs;\n                          default:-1;\n                          -1 = Unespecified;\n                          0  = None;\n                          1  = Mean;\n                          2  = CLS;\n                          3  = Last;\n                          4  = Rank)\x1b[0m: ";
    pub const params_attention_type: []const u8 = "Attention type \x1b[38;5;245m(how the model processes and relates tokens;\n                            default:-1;\n                            -1 = Unespecified;\n                            0  = Masked Self-Attention;\n                            1  = Full Self-Attention)\x1b[0m: ";
    pub const params_rope_frequency_base: []const u8 = "RoPE frequency base \x1b[38;5;245m(mathematical base for RoPE; default:0.0)\x1b[0m: ";
    pub const params_rope_frequency_scale: []const u8 = "RoPE frequency scale \x1b[38;5;245m(scaling factor to compensate for context length in RoPE; default:0.0)\x1b[0m: ";
    pub const params_yarn_extension_factor: []const u8 = "YaRN extension factor \x1b[38;5;245m(how much the context can be dynamically extended; default:-1.0)\x1b[0m: ";
    pub const params_yarn_attention_factor: []const u8 = "YaRN attention factor \x1b[38;5;245m(smoothness of attention decay at distant positions; default:1.0)\x1b[0m: ";
    pub const params_yarn_beta_fast: []const u8 = "YaRN beta fast \x1b[38;5;245m(adaptation rate to new lengths, fast initial response; default:32.0)\x1b[0m: ";
    pub const params_yarn_beta_slow: []const u8 = "YaRN beta slow \x1b[38;5;245m(adaptation rate to new lengths, fine adjustment afterward; default:1.0)\x1b[0m: ";
    pub const params_yarn_original_context: []const u8 = "YaRN original context \x1b[38;5;245m(context length the model was trained for; default:0)\x1b[0m: ";
    pub const params_defrag_threshold: []const u8 = "Defragmentation threshold \x1b[38;5;245m(when the KV cache memory is reorganized; default:-1.0)\x1b[0m: ";
    pub const params_all_logits_enabled: []const u8 = "All logits enabled \x1b[38;5;245m(full logits for all vocab tokens at each step; default:n) (y/n)\x1b[0m: ";
    pub const params_embeddings_enabled: []const u8 = "Embeddings enabled (\x1b[38;5;245moutputs embeddings from the layers; default:n) (y/n)\x1b[0m: ";
    pub const params_offload_kqv_enabled: []const u8 = "Offload KQV enabled \x1b[38;5;245m(offloads KQV matrices to the GPU during attention; default:y) (y/n)\x1b[0m: ";
    pub const params_flash_attention_enabled: []const u8 = "Flash Attention enabled \x1b[38;5;245m(optimized attention mechanism using \x1b[32mCUDA\x1b[0m\x1b[38;5;245m; default:n) (y/n)\x1b[0m: ";
    pub const params_no_performance_optimizations: []const u8 = "No performance optimizations \x1b[38;5;245m(default:y) (y/n)\x1b[0m: ";
    pub const params_rope_scaling_type: []const u8 = "RoPe scaling type \x1b[38;5;245m(how RoPE are adjusted to handle longer sequences;\n                               -1 = Unspecified;\n                               0  = None;\n                               1  = Linear;\n                               2  = YaRN;\n                               3  = Long RoPe;\n                               4  = Max Value)\x1b[0m: ";    
    pub const params_key_type: []const u8 = "Key type \x1b[38;5;245m(default:1;\n                      0 = F32;\n                      1 = F16;\n                      30 = BF16;\n                      12 = Q4_K;\n                      8 = Q8_0)\x1b[0m: ";
    pub const params_value_type: []const u8 = "Value type \x1b[38;5;245m(default:1;\n                        0 = F32;\n                        1 = F16;\n                        30 = BF16;\n                        12 = Q4_K;\n                        8 = Q8_0)\x1b[0m: ";
    // Sampling Params
    pub const sampling_min_p: []const u8 = "Add Min P Sampling? \x1b[38;5;245m(y/n)\x1b[0m ";
    pub const sampling_min_p_p: []const u8 = "                     \x1b[38;5;245mP value (float):\x1b[0m ";
    pub const sampling_min_p_min_keep: []const u8 = "                     \x1b[38;5;245mMin Keep value (usize):\x1b[0m ";
    pub const sampling_temperature: []const u8 = "Add Temperature Sampling? \x1b[38;5;245m(y/n)\x1b[0m ";
    pub const sampling_temperature_temp: []const u8 = "                           \x1b[38;5;245mTemp value (float):\x1b[0m ";
    pub const sampling_distribution: []const u8 = "Add Distribution Sampling? \x1b[38;5;245m(y/n)\x1b[0m ";
    pub const sampling_distribution_seed: []const u8 = "                            \x1b[38;5;245mSeed value (u32):\x1b[0m ";
    pub const sampling_greedy_decoding: []const u8 = "Add Greedy Decoding Sampling? \x1b[38;5;245m(y/n)\x1b[0m ";
    pub const sampling_top_k: []const u8 = "Add Top K Sampling? \x1b[38;5;245m(y/n)\x1b[0m ";
    pub const sampling_top_k_k: []const u8 = "                     \x1b[38;5;245mK value (i32):\x1b[0m ";
    pub const sampling_top_p: []const u8 = "Add Top P Sampling? \x1b[38;5;245m(y/n)\x1b[0m ";
    pub const sampling_top_p_p: []const u8 = "                     \x1b[38;5;245mP value (float):\x1b[0m ";
    pub const sampling_top_p_min_keep: []const u8 = "                     \x1b[38;5;245mMin Keep value (usize):\x1b[0m ";
    pub const sampling_typical: []const u8 = "Add Typical Sampling? \x1b[38;5;245m(y/n)\x1b[0m ";
    pub const sampling_typical_p: []const u8 = "                       \x1b[38;5;245mP value (float):\x1b[0m ";
    pub const sampling_typical_min_keep: []const u8 = "                       \x1b[38;5;245mMin Keep value (usize):\x1b[0m ";
    pub const sampling_temperature_advanced: []const u8 = "Add Advanced Temperature Sampling? \x1b[38;5;245m(y/n)\x1b[0m ";
    pub const sampling_temperature_advanced_temp: []const u8 = "                                    \x1b[38;5;245mTemp value (float)\x1b[0m: ";
    pub const sampling_temperature_advanced_delta: []const u8 = "                                    \x1b[38;5;245mDelta value (float):\x1b[0m ";
    pub const sampling_temperature_advanced_exponent: []const u8 = "                                    \x1b[38;5;245mExponent value (float):\x1b[0m ";
    pub const sampling_typical_controlled: []const u8 = "Add Typical Controled Sampling? \x1b[38;5;245m(y/n)\x1b[0m ";
    pub const sampling_typical_controlled_p: []const u8 = "                                 \x1b[38;5;245mP value (float):\x1b[0m ";
    pub const sampling_typical_controlled_temp: []const u8 = "                                 \x1b[38;5;245mTemp value (float):\x1b[0m ";
    pub const sampling_typical_controlled_min_keep: []const u8 = "                                 \x1b[38;5;245mMin Keep value (usize):\x1b[0m ";
    pub const sampling_typical_controlled_seed: []const u8 = "                                 \x1b[38;5;245mSeed value (u32):\x1b[0m ";
    pub const sampling_standard_deviation: []const u8 = "Add Standard Deviation Sampling? \x1b[38;5;245m(y/n)\x1b[0m ";
    pub const sampling_standard_deviation_width: []const u8 = "                                  \x1b[38;5;245mWidth value (float):\x1b[0m ";
    pub const sampling_mirostat: []const u8 = "Add Mirostat Sampling? \x1b[38;5;245m(y/n)\x1b[0m ";
    pub const sampling_mirostat_seed: []const u8 = "                        \x1b[38;5;245mSeed value (u32):\x1b[0m ";
    pub const sampling_mirostat_target_surprise: []const u8 = "                        \x1b[38;5;245mTarget Surprise value (float):\x1b[0m ";
    pub const sampling_mirostat_learning_rate: []const u8 = "                        \x1b[38;5;245mLearning Rate value (float):\x1b[0m ";
    pub const sampling_mirostat_window_size: []const u8 = "                        \x1b[38;5;245mWindow Size value (i32):\x1b[0m ";
    pub const sampling_simplified_mirostat: []const u8 = "Add Simplified Mirostat Sampling? \x1b[38;5;245m(y/n)\x1b[0m ";
    pub const sampling_simplified_mirostat_seed: []const u8 = "                                   \x1b[38;5;245mSeed value (u32):\x1b[0m ";
    pub const sampling_simplified_mirostat_target_surprise: []const u8 = "                                   \x1b[38;5;245mTarget Surprise value (float):\x1b[0m ";
    pub const sampling_simplified_mirostat_learning_rate: []const u8 = "                                   \x1b[38;5;245mLearning Rate value (float):\x1b[0m ";
    pub const sampling_penalties: []const u8 = "Add Penalties Sampling? \x1b[38;5;245m(y/n)\x1b[0m ";
    pub const sampling_penalties_last_window: []const u8 = "                         \x1b[38;5;245mPenalty Last Window value (i32):\x1b[0m ";
    pub const sampling_penalties_repeat: []const u8 = "                         \x1b[38;5;245mPenalty Repeat value (float):\x1b[0m ";
    pub const sampling_penalties_frequency: []const u8 = "                         \x1b[38;5;245mPenalty Frequency value (float):\x1b[0m ";
    pub const sampling_penalties_present: []const u8 = "                         \x1b[38;5;245mPenalty Present value (float):\x1b[0m ";
    pub const sampling_infill_mode: []const u8 = "Add Infill Mode Sampling? \x1b[38;5;245m(y/n)\x1b[0m ";
    pub const sampling_dry: []const u8 = "Add Dry Sampling? \x1b[38;5;245m(y/n)\x1b[0m ";
    pub const sampling_dry_train_context_size: []const u8 = "                   \x1b[38;5;245mTrain context size value (i32):\x1b[0m ";
    pub const sampling_dry_multiplier: []const u8 = "                   \x1b[38;5;245mMultiplier value (f32):\x1b[0m ";
    pub const sampling_dry_base: []const u8 = "                   \x1b[38;5;245mBase value (f32):\x1b[0m ";
    pub const sampling_dry_allowed_length: []const u8 = "                   \x1b[38;5;245mAllowed length value (i32):\x1b[0m ";
    pub const sampling_dry_penality_last_window: []const u8 = "                   \x1b[38;5;245mPenality last window size value (i32):\x1b[0m ";
    pub const sampling_dry_breakers: []const u8 = "                   \x1b[38;5;245mBreakers (white-space separated):\x1b[0m ";
    // Chat initing
    pub const initing_message: []const u8 = "\nInitializing chat session...\n\n";
    pub const logo: []const u8 =
    \\                                                                       
    \\                _     _     __  __  ____ _           _                 
    \\            ___| |   | |   |  \\/  |/ ___| |__   __ _| |_              
    \\           |_  / |   | |   | |\\/| | |   | '_ \\ / _` | __|            
    \\            / /| |___| |___| |  | | |___| | | | (_| | |_               
    \\           /___|_____|_____|_|  |_|\\____|_| |_|\\__,_|\\__|  v0.001   
    ;
    pub const logo_yellow: []const u8 =
    "                                                                       \n" ++
    "                \x1b[1;33m_     _     __  __  ____ _           _\x1b[0m                 \n" ++
    "            \x1b[1;33m___| |   | |   |  \\/  |/ ___| |__   __ _| |_\x1b[0m              \n" ++
    "           \x1b[1;33m|_  / |   | |   | |\\/| | |   | '_ \\ / _` | __|\x1b[0m            \n" ++
    "            \x1b[1;33m/ /| |___| |___| |  | | |___| | | | (_| | |_ \x1b[0m              \n" ++
    "           \x1b[1;33m/___|_____|_____|_|  |_|\\____|_| |_|\\__,_|\\__|  v0.001\x1b[0m   \n"
    ;
    pub const slogan: []const u8 =
    \\  ╔═══════════════════════════════════════════════════════════════╗  
    \\  ║      A Zig-powered CLI to Run Local LLMs from Terminal ⚡     ║  
    \\  ╚═══════════════════════════════════════════════════════════════╝  
    ;
    pub const contact: []const u8 =
    \\    ┌──────────────────────────────────────────────────────────┐     
    \\    │  Developed by Amadeus Ferro                              │     
    \\    │                                                          │     
    \\    │  🔗 LinkedIn: https://www.linkedin.com/in/amadeusferro/  │     
    \\    │  📸 Instagram: https://www.instagram.com/amadeus.ferro/  │     
    \\    │  🗂️ Github: https://github.com/amadeusferro/             │     
    \\    │  ✉️ Email: contact.amadeusferro@gmail.com                │     
    \\    └──────────────────────────────────────────────────────────┘     
    \\
    ;
    pub const contact_aquamarine: []const u8 =
    "    ┌──────────────────────────────────────────────────────────┐     \n" ++
    "    │  Developed by Amadeus Ferro                              │     \n" ++
    "    │                                                          │     \n" ++
    "    │  🔗 LinkedIn: \x1b[36mhttps://www.linkedin.com/in/amadeusferro/\x1b[0m  │     \n" ++
    "    │  📸 Instagram: \x1b[36mhttps://www.instagram.com/amadeus.ferro/\x1b[0m  │     \n" ++
    "    │  🗂️ Github: \x1b[36mhttps://github.com/amadeusferro/\x1b[0m              │     \n" ++
    "    │  ✉️ Email: \x1b[36mcontact.amadeusferro@gmail.com\x1b[0m                 │     \n" ++
    "    └──────────────────────────────────────────────────────────┘     \n"
    ;
    // Error handling
    pub const error_StringDisplayingFailed: []const  u8 = "\x1b[38;2;255;80;80mError in string displaying\x1b[0m";
    pub const error_NumberDisplayingFailed: []const  u8 = "\x1b[38;2;255;80;80mError in number displaying\x1b[0m";
    pub const error_AddNullTerminatorFailed: []const  u8 = "\x1b[38;2;255;80;80mError in adding null terminator to string\x1b[0m";
    pub const error_AddToHistoryFailed: []const u8 = "\x1b[38;2;255;80;80mError in adding string to history\x1b[0m";
    pub const error_AddToSamplingParamsFailed: []const u8 = "\x1b[38;2;255;80;80mError in adding sampler to sampling params\x1b[0m";
    pub const error_InvalidModelFormat: []const u8 = "\x1b[38;2;255;80;80mInvalid model format\x1b[0m";
    pub const error_FileReadFailed: []const u8 = "\x1b[38;2;255;80;80mInvalid Model path\x1b[0m";
    pub const error_ModelFileLoadFailed: []const u8 = "\x1b[38;2;255;80;80mUnable to load model\x1b[0m";
    pub const error_VocabAccessFailed: []const u8 = "\x1b[38;2;255;80;80mUnable to load model vocab\x1b[0m";
    pub const error_ContextCreationFailed: []const u8 = "\x1b[38;2;255;80;80mUnable to create the llama_context\x1b[0m";
    pub const error_TokenizationFailed: []const u8 = "\x1b[38;2;255;80;80mUnable to tokenize prompt\x1b[0m";
    pub const error_ContextFull: []const u8 = "\x1b[38;2;255;80;80mContext is already full\x1b[0m";
    pub const error_DecodingFailed: []const u8 = "\x1b[38;2;255;80;80mUnable to decode the token batch\x1b[0m";
    pub const error_TokenToStringFailed: []const u8 = "\x1b[38;2;255;80;80mUnable to convert token to piece\x1b[0m";
    pub const error_TokenBufferInitFailed: []const u8 = "\x1b[38;2;255;80;80mFailed to initialize token buffer\x1b[0m";
    pub const error_FormattedBufferInitFailed: []const u8 = "\x1b[38;2;255;80;80mFailed to initialize formatted prompt buffer\x1b[0m";
    pub const error_MessageListUpdateFailed: []const u8 = "\x1b[38;2;255;80;80mFailed to update message history\x1b[0m";
    pub const error_MessageInitFailed: []const u8 = "\x1b[38;2;255;80;80mFailed to initialize chat message\x1b[0m";
    pub const error_TemplateBufferResizeFailed: []const u8 = "\x1b[38;2;255;80;80mFailed to resize template buffer\x1b[0m";
    pub const error_ResponseBufferOverflow: []const u8 = "\x1b[38;2;255;80;80mResponse buffer capacity exceeded\x1b[0m";
    pub const error_ResponseGenerationFailed: []const u8 = "\x1b[38;2;255;80;80mFailed to generate response\x1b[0m";
    pub const error_GeneratedResponseTransferFailed: []const u8 = "\x1b[38;2;255;80;80mFailed to transfer ownership of generated response\x1b[0m";
    pub const error_Unknown: []const u8 = "\x1b[38;2;255;80;80mSomething wrong occurred\x1b[0m";
    pub const error_DupeTokenFailed: []const u8 = "\x1b[38;2;255;80;80mFailed to dupe a token\x1b[0m";
    pub const error_AddToBreakersFailed: []const u8 = "\x1b[38;2;255;80;80mFailed to add to breakers list\x1b[0m";
    pub const error_AllocationFailed: []const u8 = "\x1b[38;2;255;80;80mFailed to allocate\x1b[0m";
    pub const error_InvalidValue: []const u8 = "\x1b[38;2;255;80;80mInvalid Value\x1b[0m";
    pub const error_InvalidJsonPath: []const u8 = "\x1b[38;2;255;80;80mInvalid Json path\x1b[0m";
    pub const error_InvalidJsonFormat: []const u8 = "\x1b[38;2;255;80;80mInvalid json format\x1b[0m";
};
