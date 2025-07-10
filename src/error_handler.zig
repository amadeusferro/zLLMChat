const Std = @import("std");

const Messages = @import("messages.zig").Messages;

const UtilError = @import("util.zig").UtilError;
const zLLMChatError = @import("zllmchat.zig").zLLMChatError;
const LLAMAError = @import("llama.zig").LLAMAError;
const CommandError = @import("commands.zig").CommandError;
const ParamsError = @import("params.zig").ParamsError;
const Util = @import("util.zig").Util;

pub const ErrorHandler = struct {
    messages_hashmap: Std.StaticStringMap([]const u8),

    pub fn init() ErrorHandler {
        return ErrorHandler{ .messages_hashmap = Std.StaticStringMap([]const u8).initComptime(.{
            .{ "StringDisplayingFailed", Messages.util_green ++ Messages.error_StringDisplayingFailed ++ Messages.breakrow },
            .{ "NumberDisplayingFailed", Messages.util_green ++ Messages.error_NumberDisplayingFailed ++ Messages.breakrow },
            .{ "AddNullTerminatorFailed", Messages.util_green ++ Messages.error_AddNullTerminatorFailed ++ Messages.breakrow },
            .{ "AddToHistoryFailed", Messages.zllmchat_yellow ++ Messages.error_AddToHistoryFailed ++ Messages.breakrow },
            .{ "FileReadFailed", Messages.zllmchat_yellow ++ Messages.error_FileReadFailed ++ Messages.breakrow },
            .{ "InvalidModelFormat", Messages.zllmchat_yellow ++ Messages.error_InvalidModelFormat ++ Messages.breakrow },
            .{ "AddToSamplingParamsFailed", Messages.zllmchat_yellow ++ Messages.error_AddToSamplingParamsFailed ++ Messages.breakrow },
            .{ "DupeTokenFailed", Messages.zllmchat_yellow ++ Messages.error_DupeTokenFailed ++ Messages.breakrow },
            .{ "AddToBreakersFailed", Messages.zllmchat_yellow ++ Messages.error_AddToBreakersFailed ++ Messages.breakrow },
            .{ "AllocationFailed", Messages.zllmchat_yellow ++ Messages.error_AllocationFailed ++ Messages.breakrow },
            .{ "ModelFileLoadFailed", Messages.zllmchat_yellow ++ Messages.error_ModelFileLoadFailed ++ Messages.breakrow },
            .{ "VocabAccessFailed", Messages.zllmchat_yellow ++ Messages.error_VocabAccessFailed ++ Messages.breakrow },
            .{ "ContextCreationFailed", Messages.zllmchat_yellow ++ Messages.error_ContextCreationFailed ++ Messages.breakrow },
            .{ "TokenizationFailed", Messages.zllmchat_yellow ++ Messages.error_TokenizationFailed ++ Messages.breakrow },
            .{ "ContextFull", Messages.zllmchat_yellow ++ Messages.error_ContextFull ++ Messages.breakrow },
            .{ "DecodingFailed", Messages.zllmchat_yellow ++ Messages.error_DecodingFailed ++ Messages.breakrow },
            .{ "TokenToStringFailed", Messages.zllmchat_yellow ++ Messages.error_TokenToStringFailed ++ Messages.breakrow },
            .{ "TokenBufferInitFailed", Messages.zllmchat_yellow ++ Messages.error_TokenBufferInitFailed ++ Messages.breakrow },
            .{ "FormattedBufferInitFailed", Messages.zllmchat_yellow ++ Messages.error_FormattedBufferInitFailed ++ Messages.breakrow },
            .{ "MessageListUpdateFailed", Messages.zllmchat_yellow ++ Messages.error_MessageListUpdateFailed ++ Messages.breakrow },
            .{ "MessageInitFailed", Messages.zllmchat_yellow ++ Messages.error_MessageInitFailed ++ Messages.breakrow },
            .{ "TemplateBufferResizeFailed", Messages.zllmchat_yellow ++ Messages.error_TemplateBufferResizeFailed ++ Messages.breakrow },
            .{ "ResponseBufferOverflow", Messages.zllmchat_yellow ++ Messages.error_ResponseBufferOverflow ++ Messages.breakrow },
            .{ "ResponseGenerationFailed", Messages.zllmchat_yellow ++ Messages.error_ResponseGenerationFailed ++ Messages.breakrow },
            .{ "GeneratedResponseTransferFailed", Messages.zllmchat_yellow ++ Messages.error_GeneratedResponseTransferFailed ++ Messages.breakrow },
            .{ "AddToHistoryError", Messages.zllmchat_yellow ++ Messages.error_AddToHistoryFailed ++ Messages.breakrow },
            .{ "Default", Messages.zllmchat_yellow ++ Messages.error_Unknown ++ Messages.breakrow },
            .{ "InvalidValue", Messages.zllmchat_yellow ++ Messages.error_InvalidValue ++ Messages.breakrow },
            .{ "InvalidJsonPath", Messages.zllmchat_yellow ++ Messages.error_InvalidJsonPath ++ Messages.breakrow },
            .{ "InvalidJsonFormat", Messages.zllmchat_yellow ++ Messages.error_InvalidJsonFormat ++ Messages.breakrow },
        }) };
    }
    
    pub fn handleErrors(self: @This(), err: anyerror) void {
        Util.display(self.messages_hashmap.get(@errorName(err)) orelse "Default");
    }
};
