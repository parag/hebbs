use std::sync::Arc;

use hebbs_storage::{ColumnFamilyName, StorageBackend};

use crate::error::Result;

/// Key separator between entity_id and timestamp in temporal CF keys.
/// 0xFF is safe because entity IDs are UTF-8 (max byte 0xF4).
const KEY_SEPARATOR: u8 = 0xFF;

/// Query ordering for temporal scans.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemporalOrder {
    /// Oldest first (ascending timestamp).
    Chronological,
    /// Newest first (descending timestamp).
    ReverseChronological,
}

/// Temporal index operating on the `temporal` column family.
///
/// Stores `(entity_id, timestamp) → memory_id` mappings for efficient
/// time-ordered recall per entity. Leverages RocksDB's sorted iteration
/// directly — no in-memory data structure needed.
///
/// ## Key encoding
///
/// `[entity_id bytes][0xFF separator][timestamp_us big-endian u64]`
///
/// ## Query complexity
///
/// | Operation | Complexity |
/// |-----------|-----------|
/// | All memories for entity | O(log n + k) |
/// | Time-windowed query | O(log n + k) |
/// | Count for entity | O(log n + k) |
///
/// where n = total keys in CF, k = result set size.
pub struct TemporalIndex {
    storage: Arc<dyn StorageBackend>,
}

impl TemporalIndex {
    pub fn new(storage: Arc<dyn StorageBackend>) -> Self {
        Self { storage }
    }

    /// Encode a temporal index key.
    ///
    /// Complexity: O(len(entity_id)).
    #[inline]
    pub fn encode_key(entity_id: &str, timestamp_us: u64) -> Vec<u8> {
        let mut key = Vec::with_capacity(entity_id.len() + 1 + 8);
        key.extend_from_slice(entity_id.as_bytes());
        key.push(KEY_SEPARATOR);
        key.extend_from_slice(&timestamp_us.to_be_bytes());
        key
    }

    /// Build the prefix for scanning all temporal entries of an entity.
    #[inline]
    pub fn encode_prefix(entity_id: &str) -> Vec<u8> {
        let mut prefix = Vec::with_capacity(entity_id.len() + 1);
        prefix.extend_from_slice(entity_id.as_bytes());
        prefix.push(KEY_SEPARATOR);
        prefix
    }

    /// Query memories for an entity within a time window.
    ///
    /// Returns memory IDs in the requested order, limited to `limit` results.
    ///
    /// - `start_us`: inclusive start timestamp (microseconds, 0 for unbounded)
    /// - `end_us`: exclusive end timestamp (microseconds, u64::MAX for unbounded)
    /// - `order`: chronological or reverse-chronological
    /// - `limit`: maximum number of results
    ///
    /// Complexity: O(log n + k) where k = min(matching entries, limit).
    pub fn query_range(
        &self,
        entity_id: &str,
        start_us: u64,
        end_us: u64,
        order: TemporalOrder,
        limit: usize,
    ) -> Result<Vec<(Vec<u8>, u64)>> {
        let start_key = Self::encode_key(entity_id, start_us);
        let end_key = Self::encode_key(entity_id, end_us);

        let entries =
            self.storage
                .range_iterator(ColumnFamilyName::Temporal, &start_key, &end_key)?;

        let prefix = Self::encode_prefix(entity_id);
        let prefix_len = prefix.len();

        let mut results: Vec<(Vec<u8>, u64)> = entries
            .into_iter()
            .filter(|(k, _)| k.starts_with(&prefix) && k.len() == prefix_len + 8)
            .map(|(k, v)| {
                let ts =
                    u64::from_be_bytes(k[prefix_len..prefix_len + 8].try_into().unwrap_or([0; 8]));
                (v, ts)
            })
            .collect();

        if order == TemporalOrder::ReverseChronological {
            results.reverse();
        }

        results.truncate(limit);
        Ok(results)
    }

    /// Query all memories for an entity in the given order.
    ///
    /// Complexity: O(log n + k).
    pub fn query_entity(
        &self,
        entity_id: &str,
        order: TemporalOrder,
        limit: usize,
    ) -> Result<Vec<(Vec<u8>, u64)>> {
        self.query_range(entity_id, 0, u64::MAX, order, limit)
    }

    /// Count memories for an entity.
    ///
    /// Complexity: O(log n + k).
    pub fn count_entity(&self, entity_id: &str) -> Result<usize> {
        let prefix = Self::encode_prefix(entity_id);
        let entries = self
            .storage
            .prefix_iterator(ColumnFamilyName::Temporal, &prefix)?;
        Ok(entries.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hebbs_storage::InMemoryBackend;

    fn test_index() -> TemporalIndex {
        TemporalIndex::new(Arc::new(InMemoryBackend::new()))
    }

    fn put_temporal(index: &TemporalIndex, entity_id: &str, timestamp_us: u64, memory_id: &[u8]) {
        let key = TemporalIndex::encode_key(entity_id, timestamp_us);
        index
            .storage
            .put(ColumnFamilyName::Temporal, &key, memory_id)
            .unwrap();
    }

    #[test]
    fn key_encoding_format() {
        let key = TemporalIndex::encode_key("customer_42", 1_700_000_000_000_000);
        assert_eq!(key.len(), "customer_42".len() + 1 + 8);
        assert_eq!(key["customer_42".len()], KEY_SEPARATOR);
    }

    #[test]
    fn keys_sort_chronologically() {
        let k1 = TemporalIndex::encode_key("entity", 100);
        let k2 = TemporalIndex::encode_key("entity", 200);
        let k3 = TemporalIndex::encode_key("entity", 300);
        assert!(k1 < k2);
        assert!(k2 < k3);
    }

    #[test]
    fn different_entities_isolated() {
        let ka = TemporalIndex::encode_key("alice", 100);
        let kb = TemporalIndex::encode_key("bob", 100);
        let prefix_a = TemporalIndex::encode_prefix("alice");
        let prefix_b = TemporalIndex::encode_prefix("bob");
        assert!(ka.starts_with(&prefix_a));
        assert!(!ka.starts_with(&prefix_b));
        assert!(kb.starts_with(&prefix_b));
        assert!(!kb.starts_with(&prefix_a));
    }

    #[test]
    fn query_chronological_order() {
        let index = test_index();
        put_temporal(&index, "user1", 300, &[3; 16]);
        put_temporal(&index, "user1", 100, &[1; 16]);
        put_temporal(&index, "user1", 200, &[2; 16]);

        let results = index
            .query_entity("user1", TemporalOrder::Chronological, 100)
            .unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].1, 100);
        assert_eq!(results[1].1, 200);
        assert_eq!(results[2].1, 300);
    }

    #[test]
    fn query_reverse_chronological_order() {
        let index = test_index();
        put_temporal(&index, "user1", 100, &[1; 16]);
        put_temporal(&index, "user1", 200, &[2; 16]);
        put_temporal(&index, "user1", 300, &[3; 16]);

        let results = index
            .query_entity("user1", TemporalOrder::ReverseChronological, 100)
            .unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].1, 300);
        assert_eq!(results[1].1, 200);
        assert_eq!(results[2].1, 100);
    }

    #[test]
    fn query_range_windowed() {
        let index = test_index();
        for ts in (100..=500).step_by(100) {
            put_temporal(&index, "entity", ts, &[(ts / 100) as u8; 16]);
        }

        let results = index
            .query_range("entity", 200, 400, TemporalOrder::Chronological, 100)
            .unwrap();
        assert_eq!(results.len(), 2); // ts=200, ts=300 (400 is exclusive)
        assert_eq!(results[0].1, 200);
        assert_eq!(results[1].1, 300);
    }

    #[test]
    fn query_with_limit() {
        let index = test_index();
        for ts in 0..100u64 {
            put_temporal(&index, "entity", ts, &[(ts & 0xFF) as u8; 16]);
        }

        let results = index
            .query_entity("entity", TemporalOrder::Chronological, 5)
            .unwrap();
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn query_empty_entity_returns_empty() {
        let index = test_index();
        let results = index
            .query_entity("nonexistent", TemporalOrder::Chronological, 100)
            .unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn entities_are_isolated() {
        let index = test_index();
        put_temporal(&index, "alice", 100, &[1; 16]);
        put_temporal(&index, "alice", 200, &[2; 16]);
        put_temporal(&index, "bob", 150, &[3; 16]);

        let alice = index
            .query_entity("alice", TemporalOrder::Chronological, 100)
            .unwrap();
        assert_eq!(alice.len(), 2);

        let bob = index
            .query_entity("bob", TemporalOrder::Chronological, 100)
            .unwrap();
        assert_eq!(bob.len(), 1);
    }

    #[test]
    fn count_entity() {
        let index = test_index();
        put_temporal(&index, "entity", 100, &[1; 16]);
        put_temporal(&index, "entity", 200, &[2; 16]);
        put_temporal(&index, "entity", 300, &[3; 16]);

        assert_eq!(index.count_entity("entity").unwrap(), 3);
        assert_eq!(index.count_entity("other").unwrap(), 0);
    }

    #[test]
    fn delete_removes_from_scan() {
        let index = test_index();
        let key = TemporalIndex::encode_key("entity", 100);
        index
            .storage
            .put(ColumnFamilyName::Temporal, &key, &[1; 16])
            .unwrap();

        assert_eq!(index.count_entity("entity").unwrap(), 1);

        index
            .storage
            .delete(ColumnFamilyName::Temporal, &key)
            .unwrap();

        assert_eq!(index.count_entity("entity").unwrap(), 0);
    }
}
