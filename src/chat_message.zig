const Std = @import("std");
const Allocator = Std.mem.Allocator;

const Util = @import("util.zig").Util;
const UtilError = @import("util.zig").UtilError;

const llama = @cImport({
    @cInclude("llama.h");
});

pub const ChatMessage = struct {

    msg: llama.llama_chat_message,
    role: [*:0]const u8,
    content: [*:0]const u8,

    pub fn init(allocator: Allocator, role: []const u8, content: []const u8) UtilError!@This() {
        const role_to_c_string = Util.addNullTerminator(allocator, role) catch |err| return err;
        const content_to_c_string = Util.addNullTerminator(allocator, content) catch |err| return err;
        return .{
            .role = role_to_c_string.ptr,
            .content = content_to_c_string.ptr,
            .msg = .{
                .role = role_to_c_string.ptr,
                .content = content_to_c_string.ptr,
            },
        };
    }

    pub fn deinit(self: *@This(), allocator: Allocator) void {
        const role_slice = Std.mem.span(self.role);
        const content_slice = Std.mem.span(self.content);
        allocator.free(role_slice);
        allocator.free(content_slice);
    }
};
