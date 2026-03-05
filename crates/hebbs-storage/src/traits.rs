use crate::error::{ColumnFamilyName, Result};

/// A single write or delete operation within an atomic batch.
#[derive(Debug, Clone)]
pub enum BatchOperation {
    Put {
        cf: ColumnFamilyName,
        key: Vec<u8>,
        value: Vec<u8>,
    },
    Delete {
        cf: ColumnFamilyName,
        key: Vec<u8>,
    },
}

/// Abstract key-value storage backend.
///
/// `hebbs-storage` exposes this trait — not a concrete RocksDB type — so that:
/// - Unit tests in `hebbs-core` use an in-memory implementation (millisecond tests, no temp dirs).
/// - The RocksDB implementation is swappable for future backends (SQLite, custom LSM).
/// - Embedded mode (Phase 9 FFI) can use RocksDB directly without the trait overhead if needed.
///
/// The trait is intentionally narrow: it covers only the operations the engine
/// needs and avoids leaking RocksDB-specific concepts (snapshots, merge operators,
/// compaction filters).
///
/// ## Complexity contracts
///
/// | Operation | Expected complexity |
/// |-----------|-------------------|
/// | `put` | O(1) amortized (LSM memtable insert) |
/// | `get` | O(log n) with bloom filter shortcut for misses |
/// | `delete` | O(1) amortized (tombstone write) |
/// | `write_batch` | O(k) where k = number of operations |
/// | `prefix_iterator` | O(log n + k) where k = matching keys |
/// | `range_iterator` | O(log n + k) where k = keys in range |
pub trait StorageBackend: Send + Sync {
    /// Write a single key-value pair to the given column family.
    fn put(&self, cf: ColumnFamilyName, key: &[u8], value: &[u8]) -> Result<()>;

    /// Read a single value by key from the given column family.
    /// Returns `Ok(None)` if the key does not exist.
    fn get(&self, cf: ColumnFamilyName, key: &[u8]) -> Result<Option<Vec<u8>>>;

    /// Delete a single key from the given column family.
    /// Succeeds silently if the key does not exist.
    fn delete(&self, cf: ColumnFamilyName, key: &[u8]) -> Result<()>;

    /// Execute multiple operations atomically.
    /// Either all operations succeed or none do.
    /// Maps to RocksDB WriteBatch in the RocksDB backend.
    fn write_batch(&self, operations: &[BatchOperation]) -> Result<()>;

    /// Iterate over all keys with the given prefix in the given column family.
    /// Returns `(key, value)` pairs in byte-sorted order.
    ///
    /// Complexity: O(log n) seek + O(k) scan where k = matching keys.
    fn prefix_iterator(
        &self,
        cf: ColumnFamilyName,
        prefix: &[u8],
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>>;

    /// Iterate over keys in `[start, end)` range in the given column family.
    /// Returns `(key, value)` pairs in byte-sorted order.
    ///
    /// Complexity: O(log n) seek + O(k) scan where k = keys in range.
    fn range_iterator(
        &self,
        cf: ColumnFamilyName,
        start: &[u8],
        end: &[u8],
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>>;

    /// Trigger manual compaction on a column family.
    /// Used after `forget()` to ensure deleted data is physically removed.
    fn compact(&self, cf: ColumnFamilyName) -> Result<()>;
}
