const Std = @import("std");
const Allocator = Std.mem.Allocator;
const BuildOptions = @import("build_options");

const ErrorHandler = @import("error_handler.zig").ErrorHandler;
const zLLMChat = @import("zllmchat.zig").zLLMChat;
const Params = @import("params.zig").Params;

pub fn checkLeak(gpa: *Std.heap.GeneralPurposeAllocator(.{})) void {
    const leak_status = gpa.deinit();
    if (leak_status == .leak) {
        Std.log.err("\x1b[33mError: Memory Leaking!\x1b[0m\n", .{});
    }
}

pub fn main() !void {
    var gpa = Std.heap.GeneralPurposeAllocator(.{}){};
    defer checkLeak(&gpa);
    const allocator = gpa.allocator();

    const error_handler = ErrorHandler.init();

    var params: Params = undefined;

    if (BuildOptions.params_from_json) {
        params = zLLMChat.assemblyParamsFromJson(allocator, "params.json") catch |err| {
            error_handler.handleErrors(err);
            return;
        };
    } else {
        params = zLLMChat.assemblyParams(allocator) catch |err| {
            error_handler.handleErrors(err);
            return;
        };
    }
    defer zLLMChat.deinitParams(allocator, params);

    var zllmchat = zLLMChat.init(allocator, params);
    defer zllmchat.deinit();

    zllmchat.runChat() catch |err| {
        error_handler.handleErrors(err);
        return;
    };
}
