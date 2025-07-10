const Std = @import("std");

pub const MinP = struct {
    p: f32,
    min_keep: usize
};

pub const Temperature = struct {
    temp: f32
};

pub const Distribution = struct {
    seed: u32
};

pub const GreedyDecoding = struct {

};

pub const TopK = struct {
    k: i32
};

pub const TopP = struct {
    p: f32,
    min_keep: usize
};

pub const Typical = struct {
    p: f32,
    min_keep: usize
};

pub const TemperatureAdvanced = struct {
    temp: f32,
    delta: f32,
    exponent: f32
};

pub const ExtremelyTypicalControlled = struct {
    p: f32,
    temp: f32,
    min_keep: usize,
    seed: u32
};

pub const StandardDeviation = struct {
    width: f32
};

pub const Mirostat = struct {
    // vocab_size from uploaded model vocab
    seed: u32,
    target_surprise: f32,
    learning_rate: f32, 
    window_size: i32 
};

pub const SimplifiedMirostat = struct {
    seed: u32,
    target_surprise: f32,
    learning_rate: f32
};

pub const Penalties = struct {
    penalty_last_window: i32,
    penalty_repeat: f32,
    penalty_frequency: f32,
    penality_present: f32,
};

pub const InfillMode = struct {
    // from uploaded model vocab
};

pub const Dry = struct {
    // from uploaded model vocab
    train_context_size: i32,
    multiplier: f32,
    base: f32,
    allowed_length: i32,
    penality_last_window: i32,
    breakers: [][*c]const u8,
    num_breakers: usize
};

pub const TypeName = enum {
    MinP,
    Temperature,
    Distribution,
    GreedyDecoding,
    TopK,
    TopP,
    Typical,
    TemperatureAdvanced,
    ExtremelyTypicalControlled,
    StandardDeviation,
    Mirostat,
    SimplifiedMirostat,
    Penalties,
    InfillMode,
    Dry,
    // TODO: Add all samplers
    //LogitBias,
    //Grammar,
    //GrammarLazy
};
