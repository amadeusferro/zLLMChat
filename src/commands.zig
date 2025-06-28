const Std = @import("std");
const Allocator = Std.mem.Allocator;

const Util = @import("util.zig").Util;
const Messages = @import("messages.zig").Messages;
const zLLMChat = @import("zllmchat.zig").zLLMChat;

pub const Command = struct {
    call: []const u8,
    
    info_message: []const u8,
    info_message_mint_green: []const u8,

    success_message: []const u8,
    error_message: ?[]const u8,

    perform: *const fn (self: *const Command, zllmchat: *zLLMChat) CommandError!void
};

//TODO add more cool commands

pub const commands = [2]Command{
    Command{
        .call =  "!exit!",
        .info_message = "    - write \"!exit!\" to exit\n",
        .info_message_mint_green = "    - write \"\x1b[38;2;140;255;170m!exit!\x1b[0m\" to exit\n",
        .success_message = "Exiting chat session...\n",
        .error_message = null,
        .perform = exitCommand,
    },
    Command{
        .call =  "!save!",
        .info_message = "    - write \"!save!\" to save as txt\n",
        .info_message_mint_green = "    - write \"\x1b[38;2;140;255;170m!save!\x1b[0m\" to save as txt\n",
        .success_message = "Chat saved as txt in zig-out/chats/\n",
        .error_message = "Error: Something went wrong. Please try to save again.\n",
        .perform = saveCommand,
    },
};

fn exitCommand(self: *const Command, zllmchat: *zLLMChat) CommandError!void {
    _ = zllmchat;
    Util.display(Messages.zllmchat_yellow);
    Util.display(self.success_message);
    Std.process.exit(0);
}

fn saveCommand(self: *const Command, zllmchat: *zLLMChat) CommandError!void {
    if (Util.saveAsTxt(zllmchat.allocator, zllmchat.history, "./zig-out/chats")) {
        Util.display(Messages.zllmchat_yellow);
        zllmchat.history.appendSlice(Messages.zllmchat) catch return CommandError.AddToHistoryError;
        Util.display(self.success_message);
        zllmchat.history.appendSlice(self.success_message) catch return CommandError.AddToHistoryError;
    } else {
        Util.display(Messages.zllmchat_yellow);
        zllmchat.history.appendSlice(Messages.zllmchat) catch return CommandError.AddToHistoryError;
        Util.display(self.error_message.?);
        zllmchat.history.appendSlice(self.error_message.?) catch return CommandError.AddToHistoryError;
    }
}

pub const CommandError = error {
    AddToHistoryError,
};
