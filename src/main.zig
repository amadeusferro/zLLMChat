const Std = @import("std");
const Allocator = Std.mem.Allocator;

const ErrorHandler = @import("error_handler.zig").ErrorHandler;
const zLLMChat = @import("zllmchat.zig").zLLMChat;

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


    // Uncomment it if you want Json version
    // const params = zLLMChat.assemblyParamsFromJson(allocator, "params.json") catch |err| {
    //     error_handler.handleErrors(err);
    //     return;
    // };
    // defer zLLMChat.deinitParams(allocator, params);

    // Uncomment it if you want CLI version
    // const params = zLLMChat.assemblyParams(allocator) catch |err| {
    //     error_handler.handleErrors(err);
    //     return;
    // };
    // defer zLLMChat.deinitParams(allocator, params);

    var zllmchat = zLLMChat.init(allocator, params);
    defer zllmchat.deinit();

    zllmchat.runChat() catch |err| {
        error_handler.handleErrors(err);
        return;
    };
}
