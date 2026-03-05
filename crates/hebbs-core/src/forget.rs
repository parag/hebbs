use crate::memory::MemoryKind;

/// Maximum memories deleted in a single `forget()` call (Principle 4: bounded everything).
pub const MAX_FORGET_BATCH_SIZE: usize = 100_000;

/// Default tombstone retention period: 90 days in microseconds.
pub const DEFAULT_TOMBSTONE_TTL_US: u64 = 90 * 24 * 3600 * 1_000_000;

/// Criteria for selecting memories to forget.
///
/// Multiple criteria combine with AND semantics: a memory must match
/// ALL specified criteria to be selected for deletion.
///
/// At least one criterion must be set; empty criteria are rejected.
pub struct ForgetCriteria {
    /// Forget specific memories by their 16-byte ULIDs.
    /// Complexity: O(k) point lookups.
    pub memory_ids: Vec<Vec<u8>>,

    /// Forget all memories belonging to this entity.
    /// Uses temporal index: O(log n + k_entity).
    pub entity_id: Option<String>,

    /// Forget memories whose `last_accessed_at` is older than this microsecond timestamp.
    /// A memory last accessed before this threshold is considered stale.
    pub staleness_threshold_us: Option<u64>,

    /// Forget memories with `access_count` below this threshold.
    pub access_count_floor: Option<u64>,

    /// Forget memories of a specific kind.
    pub memory_kind: Option<MemoryKind>,

    /// Forget memories with `decay_score` below this threshold.
    pub decay_score_floor: Option<f32>,
}

impl ForgetCriteria {
    /// Create criteria to forget specific memories by ID.
    pub fn by_ids(ids: Vec<Vec<u8>>) -> Self {
        Self {
            memory_ids: ids,
            entity_id: None,
            staleness_threshold_us: None,
            access_count_floor: None,
            memory_kind: None,
            decay_score_floor: None,
        }
    }

    /// Create criteria to forget all memories for an entity.
    pub fn by_entity(entity_id: impl Into<String>) -> Self {
        Self {
            memory_ids: Vec::new(),
            entity_id: Some(entity_id.into()),
            staleness_threshold_us: None,
            access_count_floor: None,
            memory_kind: None,
            decay_score_floor: None,
        }
    }

    /// Returns true if no criteria are set.
    pub fn is_empty(&self) -> bool {
        self.memory_ids.is_empty()
            && self.entity_id.is_none()
            && self.staleness_threshold_us.is_none()
            && self.access_count_floor.is_none()
            && self.memory_kind.is_none()
            && self.decay_score_floor.is_none()
    }

    /// Returns true if only explicit IDs are specified (fast path).
    pub fn is_id_only(&self) -> bool {
        !self.memory_ids.is_empty()
            && self.entity_id.is_none()
            && self.staleness_threshold_us.is_none()
            && self.access_count_floor.is_none()
            && self.memory_kind.is_none()
            && self.decay_score_floor.is_none()
    }
}

/// Result of a `forget()` operation.
#[derive(Debug)]
pub struct ForgetOutput {
    /// Number of memories successfully forgotten.
    pub forgotten_count: usize,
    /// Number of predecessor snapshots cascade-deleted.
    pub cascade_count: usize,
    /// Whether more candidates remain beyond the batch limit.
    pub truncated: bool,
    /// Tombstone keys written to meta CF.
    pub tombstone_count: usize,
}

/// Configuration for the forget operation.
#[derive(Debug, Clone)]
pub struct ForgetConfig {
    /// Maximum memories deleted in a single `forget()` call.
    /// Bounded at [1, 100_000]. Default: 1000.
    pub max_batch_size: usize,

    /// How long tombstones are retained in meta CF (microseconds).
    /// Default: 90 days.
    pub tombstone_ttl_us: u64,

    /// Whether forget cascades to revision predecessor snapshots.
    /// Default: true.
    pub cascade_snapshots: bool,

    /// Whether forget triggers async compaction on affected CFs.
    /// Default: true.
    pub trigger_compaction: bool,
}

impl Default for ForgetConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 1_000,
            tombstone_ttl_us: DEFAULT_TOMBSTONE_TTL_US,
            cascade_snapshots: true,
            trigger_compaction: true,
        }
    }
}

/// A tombstone record written to the meta CF after a memory is forgotten.
///
/// Captures the audit trail required for GDPR proof-of-deletion
/// and Phase 7 insight invalidation.
///
/// Keyed in meta CF as: `tombstone:[timestamp_us BE 8B]:[memory_id 16B]`
#[derive(Debug, Clone, bitcode::Encode, bitcode::Decode)]
pub struct Tombstone {
    /// The forgotten memory's ULID.
    pub memory_id: Vec<u8>,
    /// Entity scope (if any).
    pub entity_id: Option<String>,
    /// When the deletion occurred (microseconds since epoch).
    pub forget_timestamp_us: u64,
    /// Description of the criteria that matched.
    pub criteria_description: String,
    /// Number of predecessor snapshots cascade-deleted.
    pub cascade_count: u32,
    /// SHA-256 hash of the content (NOT the content itself).
    pub content_hash: Vec<u8>,
}

impl Tombstone {
    pub fn to_bytes(&self) -> Vec<u8> {
        bitcode::encode(self)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        bitcode::decode(bytes).map_err(|e| format!("tombstone deserialization failed: {}", e))
    }
}

/// Encode a tombstone key for the meta CF.
///
/// Format: `tombstone:[timestamp_us BE 8B]:[memory_id 16B]`
/// This enables range-scanning tombstones by time for garbage collection.
pub fn encode_tombstone_key(timestamp_us: u64, memory_id: &[u8]) -> Vec<u8> {
    let prefix = b"tombstone:";
    let mut key = Vec::with_capacity(prefix.len() + 8 + 1 + memory_id.len());
    key.extend_from_slice(prefix);
    key.extend_from_slice(&timestamp_us.to_be_bytes());
    key.push(b':');
    key.extend_from_slice(memory_id);
    key
}

/// Prefix for scanning all tombstones.
pub fn tombstone_prefix() -> Vec<u8> {
    b"tombstone:".to_vec()
}
