use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::memory::Memory;
use hebbs_index::EdgeType;

/// Maximum top_k value for recall operations (Principle 4: bounded everything).
pub const MAX_TOP_K: usize = 1000;

/// Maximum graph traversal depth (Principle 4, ScalabilityArchitecture.md).
pub const MAX_TRAVERSAL_DEPTH: usize = 10;

/// Maximum memories returned by prime() (Principle 4).
pub const MAX_PRIME_MEMORIES: usize = 200;

/// Default recency window for prime() in microseconds (7 days).
pub const DEFAULT_PRIME_RECENCY_WINDOW_US: u64 = 7 * 24 * 3600 * 1_000_000;

/// Default max_age for recency scoring in microseconds (30 days).
pub const DEFAULT_MAX_AGE_US: u64 = 30 * 24 * 3600 * 1_000_000;

/// Default reinforcement cap for logarithmic scaling.
pub const DEFAULT_REINFORCEMENT_CAP: u64 = 100;

/// Recall strategy variants. The caller explicitly selects the retrieval mode.
///
/// Each strategy is an independent retrieval path with its own latency profile,
/// index dependency, and relevance signal.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RecallStrategy {
    /// Embed the cue, query HNSW, return top-K ranked by embedding distance.
    /// Complexity: O(embed) + O(log n * ef_search) + O(k * point_lookup).
    Similarity,

    /// Query temporal index by entity_id and time range.
    /// Complexity: O(log n + k).
    Temporal,

    /// Find seed memory (by ID or embedding closest match), traverse graph edges.
    /// Complexity: O(embed or point_lookup) + O(branching_factor^max_depth) + O(k * point_lookup).
    Causal,

    /// Embed cue, query HNSW with wider search, re-rank by composite
    /// embedding + structural similarity.
    /// Complexity: O(embed) + O(log n * 2 * ef_search) + O(candidates * structural_compare).
    Analogical,
}

/// Direction for causal graph traversal using the associative HNSW.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum CausalDirection {
    /// Follow edges from cause to effect (forward direction).
    Forward,
    /// Follow edges from effect to cause (backward direction).
    Backward,
    /// Search both directions and merge results.
    #[default]
    Both,
}

/// Input for the `recall()` operation.
///
/// The cue is required. Strategy selection is explicit (Principle 3:
/// cognition, not storage). The caller decides the retrieval mode.
pub struct RecallInput {
    /// The query signal. Embedded for similarity/analogical strategies,
    /// parsed for entity_id in temporal/causal strategies.
    /// Max length: same as content (64KB).
    pub cue: String,

    /// One or more strategies to execute. When multiple strategies are
    /// specified, they run in parallel and results are merged.
    pub strategies: Vec<RecallStrategy>,

    /// Maximum results per strategy before merge. Defaults to 10.
    /// Bounded at MAX_TOP_K (1000).
    pub top_k: Option<usize>,

    /// Required for temporal strategy, optional hint for others.
    pub entity_id: Option<String>,

    /// Constrains temporal strategy to a time window (start_us, end_us).
    pub time_range: Option<(u64, u64)>,

    /// Constrains causal strategy to specific edge types.
    pub edge_types: Option<Vec<EdgeType>>,

    /// Bounds causal graph traversal depth. Defaults to 5.
    /// Hard maximum: MAX_TRAVERSAL_DEPTH (10).
    pub max_depth: Option<usize>,

    /// Override HNSW ef_search for this query. Trades latency for recall quality.
    pub ef_search: Option<usize>,

    /// Composite scoring weight overrides. When None, defaults are used.
    pub scoring_weights: Option<ScoringWeights>,

    /// Structured context for the cue. Used by analogical strategy for
    /// structural similarity comparison.
    pub cue_context: Option<HashMap<String, serde_json::Value>>,

    /// Direction for causal traversal. Defaults to Both (bidirectional).
    pub causal_direction: Option<CausalDirection>,

    /// Memory A for analogical recall (A:B::C:?). Enables vector arithmetic.
    pub analogy_a_id: Option<[u8; 16]>,

    /// Memory B for analogical recall (A:B::C:?). Enables vector arithmetic.
    pub analogy_b_id: Option<[u8; 16]>,
}

impl RecallInput {
    /// Create a simple recall input with a single strategy.
    pub fn new(cue: impl Into<String>, strategy: RecallStrategy) -> Self {
        Self {
            cue: cue.into(),
            strategies: vec![strategy],
            top_k: None,
            entity_id: None,
            time_range: None,
            edge_types: None,
            max_depth: None,
            ef_search: None,
            scoring_weights: None,
            cue_context: None,
            causal_direction: None,
            analogy_a_id: None,
            analogy_b_id: None,
        }
    }

    /// Create a multi-strategy recall input.
    pub fn multi(cue: impl Into<String>, strategies: Vec<RecallStrategy>) -> Self {
        Self {
            cue: cue.into(),
            strategies,
            top_k: None,
            entity_id: None,
            time_range: None,
            edge_types: None,
            max_depth: None,
            ef_search: None,
            scoring_weights: None,
            cue_context: None,
            causal_direction: None,
            analogy_a_id: None,
            analogy_b_id: None,
        }
    }
}

/// Configurable weights for composite scoring.
///
/// All signals are normalized to [0.0, 1.0] before weighting.
/// Default weights emphasize relevance (0.5) with recency (0.2),
/// importance (0.2), and reinforcement (0.1) as secondary signals.
#[derive(Debug, Clone, Copy)]
pub struct ScoringWeights {
    /// Weight for strategy-specific relevance signal. Default: 0.5.
    pub w_relevance: f32,
    /// Weight for temporal recency. Default: 0.2.
    pub w_recency: f32,
    /// Weight for stored importance. Default: 0.2.
    pub w_importance: f32,
    /// Weight for access-count reinforcement. Default: 0.1.
    pub w_reinforcement: f32,
    /// Maximum age for recency scoring in microseconds. Default: 30 days.
    pub max_age_us: u64,
    /// Cap for reinforcement logarithmic scaling. Default: 100.
    pub reinforcement_cap: u64,
}

impl Default for ScoringWeights {
    fn default() -> Self {
        Self {
            w_relevance: 0.5,
            w_recency: 0.2,
            w_importance: 0.2,
            w_reinforcement: 0.1,
            max_age_us: DEFAULT_MAX_AGE_US,
            reinforcement_cap: DEFAULT_REINFORCEMENT_CAP,
        }
    }
}

/// Configurable weights for analogical structural similarity.
#[derive(Debug, Clone, Copy)]
pub struct AnalogicalWeights {
    /// Balance between embedding similarity and structural similarity.
    /// 1.0 = pure embedding, 0.0 = pure structural. Default: 0.5.
    pub alpha: f32,
    /// Weight for context key overlap in structural score. Default: 0.4.
    pub key_overlap_weight: f32,
    /// Weight for value type matching. Default: 0.3.
    pub value_type_match_weight: f32,
    /// Weight for MemoryKind matching. Default: 0.2.
    pub kind_match_weight: f32,
    /// Weight for entity pattern matching. Default: 0.1.
    pub entity_pattern_weight: f32,
}

impl Default for AnalogicalWeights {
    fn default() -> Self {
        Self {
            alpha: 0.5,
            key_overlap_weight: 0.4,
            value_type_match_weight: 0.3,
            kind_match_weight: 0.2,
            entity_pattern_weight: 0.1,
        }
    }
}

/// A single result from a recall operation.
#[derive(Debug, Clone)]
pub struct RecallResult {
    /// The recalled memory.
    pub memory: Memory,
    /// Composite score combining relevance, recency, importance, and reinforcement.
    /// Range: [0.0, sum_of_weights].
    pub score: f32,
    /// Which strategies found this memory and strategy-specific signals.
    pub strategy_details: Vec<StrategyDetail>,
}

/// Strategy-specific metadata for a recall result.
#[derive(Debug, Clone)]
pub enum StrategyDetail {
    Similarity {
        /// L2 distance from the query embedding (lower = more similar).
        distance: f32,
        /// Relevance score: 1.0 - distance, in [0.0, 1.0].
        relevance: f32,
    },
    Temporal {
        /// Timestamp of the memory.
        timestamp: u64,
        /// Rank position in the temporal result set.
        rank: usize,
        /// Relevance from rank: 1.0 - (rank / top_k).
        relevance: f32,
    },
    Causal {
        /// Graph distance from the seed memory.
        depth: usize,
        /// Edge type that connected this memory.
        edge_type: EdgeType,
        /// Memory ID of the traversal seed.
        seed_id: [u8; 16],
        /// Relevance from depth: 1.0 - (depth / max_depth).
        relevance: f32,
    },
    Analogical {
        /// Embedding similarity component.
        embedding_similarity: f32,
        /// Structural similarity component.
        structural_similarity: f32,
        /// Combined analogical relevance.
        relevance: f32,
        /// Whether the result was found via vector arithmetic (assoc HNSW).
        used_vector_analogy: bool,
    },
}

impl StrategyDetail {
    /// Extract the relevance score from any strategy detail variant.
    pub fn relevance(&self) -> f32 {
        match self {
            StrategyDetail::Similarity { relevance, .. } => *relevance,
            StrategyDetail::Temporal { relevance, .. } => *relevance,
            StrategyDetail::Causal { relevance, .. } => *relevance,
            StrategyDetail::Analogical { relevance, .. } => *relevance,
        }
    }

    /// Which strategy produced this detail.
    pub fn strategy(&self) -> RecallStrategy {
        match self {
            StrategyDetail::Similarity { .. } => RecallStrategy::Similarity,
            StrategyDetail::Temporal { .. } => RecallStrategy::Temporal,
            StrategyDetail::Causal { .. } => RecallStrategy::Causal,
            StrategyDetail::Analogical { .. } => RecallStrategy::Analogical,
        }
    }
}

/// Per-strategy result used internally before merge.
pub(crate) struct StrategyResult {
    pub memory: Memory,
    pub relevance: f32,
    pub detail: StrategyDetail,
}

/// Per-strategy execution outcome (may succeed or fail independently).
pub(crate) enum StrategyOutcome {
    Ok(Vec<StrategyResult>),
    Err(RecallStrategy, String),
}

/// Error details for a strategy that failed during multi-strategy recall.
#[derive(Debug, Clone)]
pub struct StrategyError {
    pub strategy: RecallStrategy,
    pub message: String,
}

/// Combined output from a recall operation including partial failure info.
#[derive(Debug)]
pub struct RecallOutput {
    /// Ranked, deduplicated results.
    pub results: Vec<RecallResult>,
    /// Strategies that failed (if any). Empty for single-strategy calls.
    pub strategy_errors: Vec<StrategyError>,
    /// Per-strategy truncation flags.
    pub truncated: HashMap<RecallStrategy, bool>,
    /// Time spent in the embedder, in microseconds. `None` when no
    /// embedding was required (e.g. temporal-only recall).
    pub embed_duration_us: Option<u64>,
}

/// Input for the `prime()` operation.
///
/// Framework-integration point: before every agent turn, the orchestration
/// framework calls `prime()` to load relevant context.
pub struct PrimeInput {
    /// Required. Prime always scopes to an entity.
    pub entity_id: String,

    /// Additional context keys for the similarity component.
    pub context: Option<HashMap<String, serde_json::Value>>,

    /// Maximum memories to return. Defaults to 20. Bounded at MAX_PRIME_MEMORIES (200).
    pub max_memories: Option<usize>,

    /// How far back the temporal component looks. Defaults to 7 days.
    pub recency_window_us: Option<u64>,

    /// Text cue for the similarity component. If not provided,
    /// the engine constructs a synthetic cue from context values.
    pub similarity_cue: Option<String>,

    /// Composite scoring weight overrides.
    pub scoring_weights: Option<ScoringWeights>,
}

impl PrimeInput {
    pub fn new(entity_id: impl Into<String>) -> Self {
        Self {
            entity_id: entity_id.into(),
            context: None,
            max_memories: None,
            recency_window_us: None,
            similarity_cue: None,
            scoring_weights: None,
        }
    }
}

/// Output from a prime() operation.
#[derive(Debug)]
pub struct PrimeOutput {
    /// Temporal results in chronological order followed by
    /// non-duplicate similarity results by relevance.
    pub results: Vec<RecallResult>,
    /// How many results came from the temporal component.
    pub temporal_count: usize,
    /// How many results came from the similarity component.
    pub similarity_count: usize,
}
