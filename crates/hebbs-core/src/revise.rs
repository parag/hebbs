use std::collections::HashMap;

use crate::engine::RememberEdge;

/// How context is updated during a revision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ContextMode {
    /// New keys are added, existing keys overwritten, absent keys preserved.
    #[default]
    Merge,
    /// The entire context is replaced with the provided map.
    Replace,
}

/// Input for the `revise()` operation.
///
/// At least one revisable field must be provided (content, importance, context,
/// entity_id, or edges). No-op revisions are rejected.
///
/// ## Field semantics
///
/// | Field | Revisable? | Effect |
/// |-------|-----------|--------|
/// | `content` | Yes | Re-embed, re-index HNSW. |
/// | `importance` | Yes | Resets `decay_score` to new importance. |
/// | `context` | Yes | Merge or replace per `context_mode`. |
/// | `entity_id` | Yes | Re-keys temporal index. |
/// | `edges` | Yes (additive) | New edges added, existing preserved. |
pub struct ReviseInput {
    /// The 16-byte ULID of the memory to revise.
    pub memory_id: Vec<u8>,

    /// New content. If provided, triggers re-embedding and HNSW re-index.
    pub content: Option<String>,

    /// New importance score in [0.0, 1.0]. Resets decay_score.
    pub importance: Option<f32>,

    /// Context updates. Behavior controlled by `context_mode`.
    pub context: Option<HashMap<String, serde_json::Value>>,

    /// How to apply context updates. Default: Merge.
    pub context_mode: ContextMode,

    /// New entity_id. Changing this re-keys the temporal index.
    pub entity_id: Option<Option<String>>,

    /// Additional edges to create (additive, no removal).
    pub edges: Vec<RememberEdge>,
}

impl ReviseInput {
    /// Create a minimal revision that updates content.
    pub fn new_content(memory_id: Vec<u8>, content: impl Into<String>) -> Self {
        Self {
            memory_id,
            content: Some(content.into()),
            importance: None,
            context: None,
            context_mode: ContextMode::default(),
            entity_id: None,
            edges: vec![],
        }
    }

    /// Returns true if no revisable fields are set.
    pub fn is_noop(&self) -> bool {
        self.content.is_none()
            && self.importance.is_none()
            && self.context.is_none()
            && self.entity_id.is_none()
            && self.edges.is_empty()
    }
}
