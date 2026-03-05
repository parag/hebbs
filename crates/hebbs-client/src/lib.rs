// Copyright 2025-2026 Parag Arora. Licensed under Apache-2.0.

mod client;
mod convert;
mod error;
mod retry;
mod types;

pub use client::{ClientBuilder, HebbsClient, SubscribeHandle};
pub use error::ClientError;
pub use retry::RetryPolicy;
pub use types::{
    ContextMode, EdgeType, ForgetCriteria, ForgetOutput, HealthStatus, InsightsFilter, Memory,
    MemoryKind, PrimeOptions, PrimeOutput, RecallOptions, RecallOutput, RecallResult,
    RecallStrategy, ReflectOutput, ReflectScope, RememberEdge, RememberOptions, ReviseOptions,
    ScoringWeights, ServingStatus, StrategyConfig, StrategyDetail, StrategyError, SubscribeOptions,
    SubscribePush,
};

pub type Result<T> = std::result::Result<T, ClientError>;
