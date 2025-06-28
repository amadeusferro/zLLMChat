const Std = @import("std");
const Allocator = Std.mem.Allocator;

const Util = @import("util.zig").Util;
const UtilError = @import("util.zig").UtilError;
const Params = @import("params.zig").Params;
const ParamsError = @import("params.zig").ParamsError;
const Command = @import("commands.zig").Command;
const CommandError = @import("commands.zig").CommandError;
const Messages = @import("messages.zig").Messages;
const commands = @import("commands.zig").commands;

const LLAMA = @import("llama.zig").LLAMA;
const LLAMAError = @import("llama.zig").LLAMAError;

pub const zLLMChat = struct {

    params: Params,
    allocator: Allocator,
    history: Std.ArrayList(u8),

    pub fn assemblyParams(allocator: Allocator) ParamsError!Params {
        return Params.init(allocator);
    }

    pub fn deinitParams(allocator: Allocator, params: Params) void {
        allocator.free(params.model_path);
        allocator.free(params.model_name);
        params.sampling_params.deinit();
    }

    pub fn init(allocator: Allocator, params: Params) @This() {
        return .{
            .params = params,
            .allocator = allocator,
            .history = Std.ArrayList(u8).init(allocator),
        };
    }

    pub fn deinit(self: @This()) void {
        self.history.deinit();
    }

    pub fn runChat(self: *@This()) (UtilError || LLAMAError || zLLMChatError || CommandError)!void {

        Util.display(Messages.initing_message);
        Std.Thread.sleep(5000000000);

        var llama = try LLAMA.init(self.allocator, self.params);
        defer llama.deinit();
    
        //Util.display(Messages.ansii_clear_console);
        Util.display(Messages.logo_yellow);
        try self.addToHistory(Messages.logo);
        try self.displayAndAddToHistory(Messages.breakrow ++
                                        Messages.slogan ++
                                        Messages.breakrow);
        Util.display(Messages.contact_aquamarine);
        try self.addToHistory(Messages.contact);
        try self.displayAndAddToHistory(Messages.breakrow);

        for (commands) |command| {
            Util.display(command.info_message_mint_green);
            try self.addToHistory(command.info_message);
        }

        while (true) {
            var prompt: ?[]const u8 = null;
            while(prompt == null) {
                Util.display(Messages.cursor_yellow);
                try self.addToHistory(Messages.cursor);
                prompt = Util.readString(self.allocator);
                Util.display(Messages.breakrow);
                try self.addToHistory(prompt.?);
                try self.addToHistory(Messages.breakrow ++ Messages.breakrow);
            }
            defer self.allocator.free(prompt.?);

            var is_command = false;
            for (commands) |command| {
                if(Std.mem.eql(u8, prompt.?, command.call)) {
                    try command.perform(&command, self);
                    is_command = true;
                }
            }
            if(is_command) continue;
        
            Util.display(Messages.ansii_purple);
            Util.display(self.params.model_name);
            Util.display(Messages.greaterthan_purple);
            try self.addToHistory(self.params.model_name);
            try self.addToHistory(Messages.greaterthan);

            const response = try llama.query(prompt.?);
            defer self.allocator.free(response);

            Util.display(Messages.breakrow);
            try self.addToHistory(response);
            try self.addToHistory(Messages.breakrow);
        }
    }

    fn addToHistory(self: *@This(), message: []const u8) zLLMChatError!void {
        self.history.appendSlice(message) catch return zLLMChatError.AddToHistoryFailed;
    }

    fn displayAndAddToHistory(self: *@This(), message: []const u8) zLLMChatError!void {
        self.addToHistory(message) catch |err| return err;
        Util.display(message);
    }
};

pub const zLLMChatError = error {
    AddToHistoryFailed,
};
