const Std = @import("std");

const Messages = @import("messages.zig").Messages;

const UtilError = @import("util.zig").UtilError;
const zLLMChatError = @import("zllmchat.zig").zLLMChatError;
const LLAMAError = @import("llama.zig").LLAMAError;
const CommandError = @import("commands.zig").CommandError;
const ParamsError = @import("params.zig").ParamsError;

pub fn handleErrors(err: anyerror) void {
    switch (err) {
        // Util Errors
        UtilError.StringDisplayingFailed => {
            Std.log.err(Messages.util_green ++ Messages.error_StringDisplayingFailed, .{});
        },
        UtilError.NumberDisplayingFailed => {
            Std.log.err(Messages.util_green ++ Messages.error_NumberDisplayingFailed, .{});
        },
        UtilError.AddNullTerminatorFailed => {
            Std.log.err(Messages.util_green ++ Messages.error_AddNullTerminatorFailed, .{});
        },
        // zLLLMChat Erros
        zLLMChatError.AddToHistoryFailed => {
            Std.log.err(Messages.zllmchat_yellow ++ Messages.error_AddToHistoryFailed, .{});
        },
        // Params Erros
        ParamsError.FileReadFailed => {
            Std.log.err(Messages.zllmchat_yellow ++ Messages.error_FileReadFailed, .{});
        },
        ParamsError.InvalidModelFormat => {
            Std.log.err(Messages.zllmchat_yellow ++ Messages.error_InvalidModelFormat, .{});
        },
        ParamsError.AddToSamplingParamsFailed => {
            Std.log.err(Messages.zllmchat_yellow ++ "Messages.error_AddToSamplingParamsFailed", .{});
        },
        // LLAMA Erros
        LLAMAError.ModelFileLoadFailed => {
            Std.log.err(Messages.zllmchat_yellow ++ Messages.error_ModelFileLoadFailed, .{});
        },
        LLAMAError.VocabAccessFailed => {
            Std.log.err(Messages.zllmchat_yellow ++ Messages.error_VocabAccessFailed, .{});
        },
        LLAMAError.ContextCreationFailed => {
            Std.log.err(Messages.zllmchat_yellow ++ Messages.error_ContextCreationFailed, .{});
        },
        LLAMAError.TokenizationFailed => {
            Std.log.err(Messages.zllmchat_yellow ++ Messages.error_TokenizationFailed, .{});
        },
        LLAMAError.ContextFull => {
            Std.log.err(Messages.zllmchat_yellow ++ Messages.error_ContextFull, .{});
        },
        LLAMAError.DecodingFailed => {
            Std.log.err(Messages.zllmchat_yellow ++ Messages.error_DecodingFailed, .{});
        },
        LLAMAError.TokenToStringFailed => {
            Std.log.err(Messages.zllmchat_yellow ++ Messages.error_TokenToStringFailed, .{});
        },
        LLAMAError.TokenBufferInitFailed => {
            Std.log.err(Messages.zllmchat_yellow ++ Messages.error_TokenBufferInitFailed, .{});
        },
        LLAMAError.FormattedBufferInitFailed => {
            Std.log.err(Messages.zllmchat_yellow ++ Messages.error_FormattedBufferInitFailed, .{});
        },
        LLAMAError.MessageListUpdateFailed => {
            Std.log.err(Messages.zllmchat_yellow ++ Messages.error_MessageListUpdateFailed, .{});
        },
        LLAMAError.MessageInitFailed => {
            Std.log.err(Messages.zllmchat_yellow ++ Messages.error_MessageInitFailed, .{});
        },
        LLAMAError.TemplateBufferResizeFailed => {
            Std.log.err(Messages.zllmchat_yellow ++ Messages.error_TemplateBufferResizeFailed, .{});
        },
        LLAMAError.ResponseBufferOverflow => {
            Std.log.err(Messages.zllmchat_yellow ++ Messages.error_ResponseBufferOverflow, .{});
        },
        LLAMAError.ResponseGenerationFailed => {
            Std.log.err(Messages.zllmchat_yellow ++ Messages.error_ResponseGenerationFailed, .{});
        },
        LLAMAError.GeneratedResponseTransferFailed => {
            Std.log.err(Messages.zllmchat_yellow ++ Messages.error_GeneratedResponseTransferFailed, .{});
        },
        // Command Erros
        CommandError.AddToHistoryError => {
            Std.log.err(Messages.zllmchat_yellow ++ Messages.error_AddToHistoryFailed, .{});
        },
        // Unknown Erros
        else => {
            Std.log.err(Messages.zllmchat_yellow ++ Messages.error_Unknown, .{});
        }
    }
}
