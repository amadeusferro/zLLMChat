const Std = @import("std");
const Allocator = Std.mem.Allocator;
const stdout = Std.io.getStdOut().writer();
const stdin = Std.io.getStdIn().reader();

const Messages = @import("messages.zig").Messages;
const ErrorHandler = @import("error_handler.zig").ErrorHandler;

pub const Util = struct {

    threadlocal var globalBuffer: [16384]u8 = [_]u8{0} ** 16384;

    const error_handler = ErrorHandler.init();

    pub fn display(message: []const u8) void {
        stdout.writeAll(message) catch error_handler.handleErrors(UtilError.StringDisplayingFailed);
    }

    pub fn displayNumber(comptime T: type, number: T) void {
        stdout.print("{}", .{number}) catch error_handler.handleErrors(UtilError.NumberDisplayingFailed);
    }

    pub fn readString(allocator: Allocator) ?[]const u8 {
        const input = stdin.readUntilDelimiter(&globalBuffer, '\n') catch "";
        const trimmed = Std.mem.trim(u8, input, " \t\n\r");
        if (trimmed.len == 0) return null;
        return allocator.dupe(u8, trimmed) catch null;
    }

    pub fn readNumber(comptime T: type, default: T) ?T {
        const input = stdin.readUntilDelimiter(&globalBuffer, '\n') catch "";
        const trimmed = Std.mem.trim(u8, input, " \t\n\r");
        if (trimmed.len == 0) return default;
        return switch(@typeInfo(T)) {
            .int, .comptime_int => Std.fmt.parseInt(T, trimmed, 10) catch null,
            .float, .comptime_float => Std.fmt.parseFloat(T, trimmed) catch null,
            else => @compileError("readNumber only supports integer and float types"),
        };
    }

    pub fn readBool(default: bool) ?bool {
        const input = stdin.readUntilDelimiter(&globalBuffer, '\n') catch "";
        const trimmed = Std.mem.trim(u8, input, " \t\n\r");
        if (trimmed.len != 1) return default;

        return switch (trimmed[0]) {
            'y', 'Y' => true,
            'n', 'N' => false,
            else => null
        };
    }

    pub fn saveAsTxt(allocator: Allocator, history: Std.ArrayList(u8), comptime dir_path: []const u8) bool {
        Std.fs.cwd().makeDir(dir_path) catch |err| {
            if (err != error.PathAlreadyExists) return false;
        };

        const timestamp = Std.time.timestamp();
        const filename = Std.fmt.bufPrint(&globalBuffer, "chat_{}.txt", .{timestamp}) catch return false;

        const path = Std.fs.path.join(allocator, &.{
            dir_path, filename
        }) catch return false;
        defer allocator.free(path);

        const file = Std.fs.cwd().createFile(path, .{}) catch return false;
        defer file.close();

        file.writeAll(history.items) catch return false;

        return true;
    }

    pub fn addNullTerminator(allocator: Allocator, str: []const u8) UtilError![:0]u8 {
        return allocator.dupeZ(u8, str) catch UtilError.AddNullTerminatorFailed;
    }

    pub fn freeNullTerminatorString(allocator: Allocator, str: [:0]u8) void {
        allocator.free(str);
    }
};

pub const UtilError = error {
    StringDisplayingFailed,
    NumberDisplayingFailed,
    AddNullTerminatorFailed,
};
