const Std = @import("std");
const Allocator = Std.mem.Allocator;

const llama = @cImport({
    @cInclude("llama.h");
});

const Util = @import("util.zig").Util;
const UtilError = @import("util.zig").UtilError;
const Params = @import("params.zig").Params;
const Sampler = @import("params.zig").Sampler;
const SamplingTypes = @import("sampling_types.zig");
const ChatMessage = @import("chat_message.zig").ChatMessage;

pub const LLAMA = struct {

    params: Params,
    allocator: Allocator,
    messages: Std.ArrayList(ChatMessage),
    model: *llama.struct_llama_model,
    context: *llama.struct_llama_context,
    sampler: [*c]llama.struct_llama_sampler,
    vocab: *const llama.struct_llama_vocab,
    formatted: Std.ArrayList(u8),
    prev_length: i32,

    threadlocal var global_buffer: [256]u8 = [_]u8{0} ** 256;

    pub fn init(allocator: Allocator, params: Params) (LLAMAError || UtilError)!@This() {

        llama.ggml_backend_load_all();

        const model_params = llama.llama_model_params {
            .n_gpu_layers = @as(i32, @intCast(params.model_params.gpu_layer_count)),
            .main_gpu = @as(i32, @intCast(params.model_params.main_gpu_index)),
            .split_mode = @as(c_uint, @intCast(@intFromEnum(params.model_params.tensor_split_mode))),
            .tensor_split = params.model_params.tensor_split_ratios,
            .vocab_only = params.model_params.vocab_only_mode,
            .use_mmap = params.model_params.memory_map_enabled,
            .use_mlock = params.model_params.memory_lock_enabled,
            .check_tensors = params.model_params.tensor_validation_enabled,
            // TODO
            .devices = null,
            .tensor_buft_overrides = null,
            .progress_callback = null,
            .progress_callback_user_data = null,
            .kv_overrides = null,
        };

        const model_path_to_c_str = try Util.addNullTerminator(allocator,params.model_path);
        defer Util.freeNullTerminatorString(allocator, model_path_to_c_str);
        const model = llama.llama_model_load_from_file(model_path_to_c_str, model_params) orelse {
            return LLAMAError.ModelFileLoadFailed;
        };

        const vocab = llama.llama_model_get_vocab(model) orelse {
            return LLAMAError.VocabAccessFailed;
        };

        const context_params = llama.struct_llama_context_params{
            .attention_type = @as(c_int, @intCast(@intFromEnum(params.context_params.attention_type))),
            .defrag_thold = params.context_params.defrag_threshold,
            .embeddings = params.context_params.embeddings_enabled,
            .flash_attn = params.context_params.flash_attention_enabled,
            .logits_all = params.context_params.all_logits_enabled,
            .n_batch = params.context_params.batch_size,
            .n_ctx = params.context_params.context_size,
            .n_seq_max = params.context_params.max_sequence_length,
            .n_threads = @as(i32, @intCast(params.context_params.thread_count)),
            .n_threads_batch = @as(i32, @intCast(params.context_params.batch_thread_count)),
            .n_ubatch = params.context_params.unified_batch_size,
            .no_perf = params.context_params.no_performance_optimizations,
            .offload_kqv = params.context_params.offload_kqv_enabled,
            .pooling_type = @as(c_int, @intCast(@intFromEnum(params.context_params.pooling_type))),
            .rope_freq_base = params.context_params.rope_frequency_base,
            .rope_freq_scale = params.context_params.rope_frequency_scale,
            .rope_scaling_type = @as(c_int, @intCast(@intFromEnum(params.context_params.rope_scaling_type))),
            .type_k = @as(c_uint, @intCast(@intFromEnum(params.context_params.key_type))),
            .type_v = @as(c_uint, @intCast(@intFromEnum(params.context_params.value_type))),
            .yarn_attn_factor = params.context_params.yarn_attention_factor,
            .yarn_beta_fast = params.context_params.yarn_beta_fast,
            .yarn_beta_slow = params.context_params.yarn_beta_slow,
            .yarn_ext_factor = params.context_params.yarn_extension_factor,
            .yarn_orig_ctx = params.context_params.yarn_original_context,
            .abort_callback = null,
            .abort_callback_data = null,
            .cb_eval = null,
            .cb_eval_user_data = null,
        };

        const context = llama.llama_init_from_model(model, context_params) orelse {
            return LLAMAError.ContextCreationFailed;
        };

        const sampler = llama.llama_sampler_chain_init(llama.llama_sampler_chain_default_params());

        for(params.sampling_params.items) |sampler_param| {
            addToChain(sampler, sampler_param);
        }

        llama.llama_sampler_chain_add(sampler, llama.llama_sampler_init_min_p(0.05, 1));
        llama.llama_sampler_chain_add(sampler, llama.llama_sampler_init_temp(0.8));
        llama.llama_sampler_chain_add(sampler, llama.llama_sampler_init_dist(llama.LLAMA_DEFAULT_SEED));

        // llama.llama_sampler_init_greedy()
        // llama.llama_sampler_init_top_k(k: i32)
        // llama.llama_sampler_init_top_p(p: f32, min_keep: usize)
        // llama.llama_sampler_init_typical(p: f32, min_keep: usize)
        // llama.llama_sampler_init_temp_ext(t: f32, delta: f32, exponent: f32)
        // llama.llama_sampler_init_xtc(p: f32, t: f32, min_keep: usize, seed: u32)
        // llama.llama_sampler_init_top_n_sigma(n: f32)
        // llama.llama_sampler_init_mirostat(n_vocab: i32, seed: u32, tau: f32, eta: f32, m: i32)
        // llama.llama_sampler_init_mirostat_v2(seed: u32, tau: f32, eta: f32)
        // llama.llama_sampler_init_grammar(vocab: ?*const struct_llama_vocab, grammar_str: [*c]const u8, grammar_root: [*c]const u8)
        // llama.llama_sampler_init_grammar_lazy_patterns(vocab: ?*const struct_llama_vocab, grammar_str: [*c]const u8, grammar_root: [*c]const u8, trigger_patterns: [*c][*c]const u8, num_trigger_patterns: usize, trigger_tokens: [*c]const llama_token, num_trigger_tokens: usize)
        // llama.llama_sampler_init_penalties(penalty_last_n: i32, penalty_repeat: f32, penalty_freq: f32, penalty_present: f32)
        // llama.llama_sampler_init_dry(vocab: ?*const struct_llama_vocab, n_ctx_train: i32, dry_multiplier: f32, dry_base: f32, dry_allowed_length: i32, dry_penalty_last_n: i32, seq_breakers: [*c][*c]const u8, num_breakers: usize)
        // llama.llama_sampler_init_logit_bias(n_vocab: i32, n_logit_bias: i32, logit_bias: [*c]const llama_logit_bias)
        // llama.llama_sampler_init_infill(vocab: ?*const struct_llama_vocab)

        return .{
            .params = params,
            .allocator = allocator,
            .messages = Std.ArrayList(ChatMessage).init(allocator),
            .model = model,
            .context = context,
            .sampler = sampler,
            .vocab = vocab,
            .formatted = Std.ArrayList(u8).initCapacity(allocator, llama.llama_n_ctx(context)) catch return LLAMAError.FormattedBufferInitFailed,
            .prev_length = 0,
        };
    }

    pub fn deinit(self: *@This()) void {
        llama.llama_sampler_free(self.sampler);
        llama.llama_free(self.context);
        llama.llama_model_free(self.model);

        for(self.messages.items) |*msg| {
            msg.deinit(self.allocator);
        }
        self.messages.deinit();
        self.formatted.deinit();
    }

    fn generate(self: *@This(), prompt: []const u8) (LLAMAError || UtilError)![]const u8 {

        var response = Std.ArrayList(u8).init(self.allocator);
        errdefer response.deinit();

        const is_first = llama.llama_kv_self_used_cells(self.context) == 0;

        const prompt_to_c_string = try Util.addNullTerminator(self.allocator,prompt);
        defer Util.freeNullTerminatorString(self.allocator, prompt_to_c_string);
        const num_prompt_tokens = -llama.llama_tokenize(
            self.vocab, 
            prompt_to_c_string.ptr, 
            @as(i32, @intCast(prompt.len)), 
            null, 
            0, 
            is_first, 
            true
        );

        var prompt_tokens = Std.ArrayList(llama.llama_token).initCapacity(self.allocator, @as(usize, @intCast(num_prompt_tokens))) catch return LLAMAError.TokenBufferInitFailed;
        defer prompt_tokens.deinit();
        //errdefer prompt_tokens.deinit();

        const result = llama.llama_tokenize(
            self.vocab, 
            prompt_to_c_string.ptr, 
            @as(i32, @intCast(prompt.len)), 
            prompt_tokens.items.ptr, 
            @as(i32, @intCast(prompt_tokens.capacity)), 
            is_first, 
            true
        );
        
        if (result < 0) {
            return LLAMAError.TokenizationFailed;
        }

        prompt_tokens.items.len = @as(usize, @intCast(result));

        var batch = llama.llama_batch_get_one(prompt_tokens.items.ptr, @as(c_int, @intCast(prompt_tokens.items.len)));
        var new_token_id: llama.llama_token = undefined;

        while (true) {

            const num_context = llama.llama_n_ctx(self.context);
            const num_context_used = llama.llama_kv_self_used_cells(self.context);

            if (num_context_used + batch.n_tokens > num_context) {
                return LLAMAError.ContextFull;
            }

            if (llama.llama_decode(self.context, batch) != 0) {
                return LLAMAError.DecodingFailed;          
            }

            new_token_id = llama.llama_sampler_sample(self.sampler, self.context, -1);

            if (llama.llama_vocab_is_eog(self.vocab, new_token_id)) {
                break;
            }

            const n = llama.llama_token_to_piece(self.vocab, new_token_id, &global_buffer, global_buffer.len, 0, true);

            if (n < 0) {
                return LLAMAError.TokenToStringFailed;
            }

            Util.display(global_buffer[0..@as(usize, @intCast(n))]);

            response.appendSlice(global_buffer[0..@as(usize, @intCast(n))]) catch return LLAMAError.ResponseBufferOverflow;

            batch = llama.llama_batch_get_one(&new_token_id, 1);
        }
        return response.toOwnedSlice() catch LLAMAError.GeneratedResponseTransferFailed;
    }

    pub fn query(self: *@This(), prompt: []const u8) ![]const u8 {

        const template = llama.llama_model_chat_template(self.model, null); 



        const user_msg = ChatMessage.init(self.allocator, "user", prompt) catch return LLAMAError.MessageInitFailed;
        self.messages.append(user_msg) catch return LLAMAError.MessageListUpdateFailed;



        var new_length = llama.llama_chat_apply_template(
            template,
            @ptrCast(&self.messages.items[self.messages.items.len - 1]),
            1,
            true,
            @ptrCast(self.formatted.items.ptr),
            @as(i32, @intCast(self.formatted.items.len)),
        );



        if (new_length > @as(i32, @intCast(self.formatted.items.len))) {
            self.formatted.resize(@as(usize, @intCast(new_length))) catch return LLAMAError.TemplateBufferResizeFailed;
            new_length = llama.llama_chat_apply_template(
                template,
                @ptrCast(&self.messages.items[self.messages.items.len - 1]),
                1,
                true,
                @ptrCast(self.formatted.items.ptr),
                @as(i32, @intCast(self.formatted.items.len)),
            );
        }



        const response = self.generate(self.formatted.items[0..@as(usize, @intCast(new_length))]) catch return LLAMAError.ResponseGenerationFailed;      



        const assistant_msg = ChatMessage.init(self.allocator, "assistant", response) catch return LLAMAError.MessageInitFailed;
        self.messages.append(assistant_msg) catch return LLAMAError.MessageListUpdateFailed;


        self.prev_length = llama.llama_chat_apply_template(
            template,
            @ptrCast(self.messages.items.ptr),
            self.messages.items.len,
            false,
            null,
            0,
        );

        return response;
    }

    fn addToChain(sampler: [*c]llama.struct_llama_sampler, sampler_param: Sampler) void {
        const param = sampler_param.getParams(sampler_param.param_type);
        
        switch (sampler_param.param_type) {
            SamplingTypes.MinP => llama.llama_sampler_chain_add(sampler, llama.llama_sampler_init_min_p(param.p, param.min_keep)),
            SamplingTypes.Temperature => llama.llama_sampler_chain_add(sampler, llama.llama_sampler_init_temp(param.temp)),
            SamplingTypes.Distribution => llama.llama_sampler_chain_add(sampler, llama.llama_sampler_init_dist(param.seed)),
        }
    }
};

pub const LLAMAError = error {
    ModelFileLoadFailed,
    VocabAccessFailed,
    ContextCreationFailed,
    TokenizationFailed,
    ContextFull,
    DecodingFailed,
    TokenToStringFailed,
    TokenBufferInitFailed,
    FormattedBufferInitFailed,
    MessageListUpdateFailed,
    MessageInitFailed,
    TemplateBufferResizeFailed,
    ResponseBufferOverflow,
    ResponseGenerationFailed,
    GeneratedResponseTransferFailed,
};
