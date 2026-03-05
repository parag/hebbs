use serde::{Deserialize, Serialize};

/// Lightweight memory representation decoupling hebbs-reflect from hebbs-core.
/// Engine converts its `Memory` struct into this before calling the pipeline.
#[derive(Debug, Clone)]
pub struct MemoryEntry {
    pub id: [u8; 16],
    pub content: String,
    pub importance: f32,
    pub entity_id: Option<String>,
    pub embedding: Vec<f32>,
    pub created_at: u64,
}

/// Input to the reflection pipeline.
#[derive(Debug)]
pub struct ReflectInput {
    pub memories: Vec<MemoryEntry>,
    pub existing_insights: Vec<MemoryEntry>,
    pub config: PipelineConfig,
}

/// Configuration for the four pipeline stages.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub min_cluster_size: usize,
    pub max_clusters: usize,
    pub clustering_seed: u64,
    pub max_iterations: usize,
    pub proposal_max_tokens: usize,
    pub validation_max_tokens: usize,
    /// Weight of source memory importance when computing insight importance.
    /// Remainder `(1.0 - weight)` is the LLM confidence weight.
    pub insight_importance_weight: f32,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            min_cluster_size: 3,
            max_clusters: 50,
            clustering_seed: 42,
            max_iterations: 50,
            proposal_max_tokens: 4000,
            validation_max_tokens: 6000,
            insight_importance_weight: 0.7,
        }
    }
}

/// A cluster of related memories produced by Stage 1.
#[derive(Debug, Clone)]
pub struct Cluster {
    pub cluster_id: usize,
    pub member_indices: Vec<usize>,
    pub centroid: Vec<f32>,
}

/// Candidate insight from Stage 2 (Proposal).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateInsight {
    pub content: String,
    pub confidence: f32,
    /// Hex-encoded 16-byte memory IDs for JSON serialization.
    pub source_memory_ids: Vec<String>,
    #[serde(default)]
    pub tags: Vec<String>,
}

/// Response wrapper for a list of candidate insights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposalResponse {
    pub insights: Vec<CandidateInsight>,
}

/// Verdict from Stage 3 (Validation).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "verdict")]
pub enum InsightVerdict {
    #[serde(rename = "accepted")]
    Accepted,
    #[serde(rename = "rejected")]
    Rejected { reason: String },
    #[serde(rename = "revised")]
    Revised { revised_content: String },
    #[serde(rename = "merged")]
    MergedWithExisting { existing_id: String },
}

/// A single validated insight from Stage 3.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatedInsightEntry {
    pub candidate_index: usize,
    #[serde(flatten)]
    pub verdict: InsightVerdict,
    pub confidence: f32,
}

/// Response wrapper for validated insights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResponse {
    pub results: Vec<ValidatedInsightEntry>,
}

/// Final output insight ready for consolidation in hebbs-core.
#[derive(Debug, Clone)]
pub struct ProducedInsight {
    pub content: String,
    pub confidence: f32,
    pub source_memory_ids: Vec<[u8; 16]>,
    pub tags: Vec<String>,
    pub cluster_id: usize,
}

/// Per-cluster processing status.
#[derive(Debug, Clone)]
pub enum ClusterStatus {
    Success { insight_count: usize },
    NoInsights,
    Failed { error: String },
}

/// Info about a cluster for centroid publication.
#[derive(Debug, Clone)]
pub struct ClusterInfo {
    pub cluster_id: usize,
    pub member_count: usize,
    pub centroid: Vec<f32>,
    pub status: ClusterStatus,
}

/// Output from the reflection pipeline.
#[derive(Debug)]
pub struct ReflectOutput {
    pub insights: Vec<ProducedInsight>,
    pub clusters: Vec<ClusterInfo>,
}
