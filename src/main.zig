const Std = @import("std");
const Allocator = Std.mem.Allocator;

const handleErrors = @import("error_handler.zig").handleErrors;
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

    const params = zLLMChat.assemblyParams(allocator) catch |err| {
        handleErrors(err);
        return;
    };
    defer zLLMChat.deinitParams(allocator, params);

    var zllmchat = zLLMChat.init(allocator, params);
    defer zllmchat.deinit();

    zllmchat.runChat() catch |err| {
        handleErrors(err);
        return;
    };
}
