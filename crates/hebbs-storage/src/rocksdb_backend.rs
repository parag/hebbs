use std::path::Path;
use std::sync::Arc;

use rocksdb::{
    BlockBasedOptions, DBWithThreadMode, MultiThreaded, Options, SliceTransform,
    WriteBatchWithTransaction,
};

use crate::error::{ColumnFamilyName, Result, StorageError};
use crate::traits::{BatchOperation, StorageBackend};

/// RocksDB-backed storage implementation.
///
/// ## Thread safety
///
/// RocksDB supports concurrent reads and writes from multiple threads.
/// The `DB` handle is wrapped in `Arc` — not `Mutex` — because RocksDB
/// serializes writes internally through its WAL.  Reads go directly to
/// the block cache / memtable with no application-level locking.
///
/// ## Column families
///
/// All five column families (`default`, `temporal`, `vectors`, `graph`, `meta`)
/// are created on first open and must be present on subsequent opens.
/// Creating them all in Phase 1 avoids migration steps in later phases.
pub struct RocksDbBackend {
    db: Arc<DBWithThreadMode<MultiThreaded>>,
}

impl RocksDbBackend {
    /// Open (or create) a RocksDB database at the given path.
    ///
    /// ## Tuning rationale (Phase 1 conservative defaults)
    ///
    /// - WAL enabled for durability.
    /// - `sync = false`: WAL is fsynced periodically, not on every write.
    ///   Acceptable because source data (agent interactions) can be replayed.
    /// - `write_buffer_size = 64 MB`: reduces write amplification.
    /// - Bloom filters: 10 bits/key on all CFs for O(1) negative lookups.
    /// - Block cache: 256 MB shared LRU — the largest memory consumer.
    /// - Level compaction with 2 background threads and 100 MB/s rate limit.
    pub fn open(data_dir: impl AsRef<Path>) -> Result<Self> {
        let path = data_dir.as_ref();

        let mut block_opts = BlockBasedOptions::default();
        block_opts.set_bloom_filter(10.0, false);
        block_opts.set_block_size(16 * 1024);
        // 256 MB shared block cache
        let cache = rocksdb::Cache::new_lru_cache(256 * 1024 * 1024);
        block_opts.set_block_cache(&cache);
        block_opts.set_cache_index_and_filter_blocks(true);
        block_opts.set_pin_l0_filter_and_index_blocks_in_cache(true);

        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);
        opts.set_block_based_table_factory(&block_opts);

        // Write path tuning
        opts.set_write_buffer_size(64 * 1024 * 1024);
        opts.set_max_write_buffer_number(3);
        opts.set_min_write_buffer_number_to_merge(1);

        // Compaction tuning
        opts.set_level_compaction_dynamic_level_bytes(true);
        opts.set_max_background_jobs(4);
        opts.set_compaction_style(rocksdb::DBCompactionStyle::Level);
        opts.set_target_file_size_base(64 * 1024 * 1024);

        // Parallelism
        opts.increase_parallelism(2);

        // Compression: LZ4 for speed on all levels except bottommost (Zstd for ratio)
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_bottommost_compression_type(rocksdb::DBCompressionType::Zstd);

        let cf_names: Vec<&str> = ColumnFamilyName::all()
            .iter()
            .map(|cf| cf.as_str())
            .collect();

        let cf_descriptors: Vec<rocksdb::ColumnFamilyDescriptor> = cf_names
            .iter()
            .map(|name| {
                let mut cf_opts = Options::default();
                cf_opts.set_block_based_table_factory(&block_opts);
                cf_opts.set_write_buffer_size(64 * 1024 * 1024);
                cf_opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
                cf_opts.set_bottommost_compression_type(rocksdb::DBCompressionType::Zstd);

                if *name == ColumnFamilyName::Temporal.as_str() {
                    // Temporal CF uses prefix-based iteration on entity_id
                    cf_opts.set_prefix_extractor(SliceTransform::create_noop());
                }

                rocksdb::ColumnFamilyDescriptor::new(*name, cf_opts)
            })
            .collect();

        let db =
            DBWithThreadMode::<MultiThreaded>::open_cf_descriptors(&opts, path, cf_descriptors)
                .map_err(|e| StorageError::Io {
                    operation: "open",
                    message: format!("failed to open RocksDB at {}: {}", path.display(), e),
                    source: Some(Box::new(e)),
                })?;

        Ok(Self { db: Arc::new(db) })
    }

    fn cf_handle(&self, cf: ColumnFamilyName) -> Result<Arc<rocksdb::BoundColumnFamily<'_>>> {
        self.db
            .cf_handle(cf.as_str())
            .ok_or_else(|| StorageError::Internal {
                operation: "cf_handle",
                message: format!(
                    "column family '{}' not found — database may be corrupted",
                    cf
                ),
            })
    }
}

impl StorageBackend for RocksDbBackend {
    fn put(&self, cf: ColumnFamilyName, key: &[u8], value: &[u8]) -> Result<()> {
        let handle = self.cf_handle(cf)?;
        self.db
            .put_cf(&handle, key, value)
            .map_err(|e| StorageError::Io {
                operation: "put",
                message: format!("write failed in CF '{}': {}", cf, e),
                source: Some(Box::new(e)),
            })
    }

    fn get(&self, cf: ColumnFamilyName, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let handle = self.cf_handle(cf)?;
        self.db.get_cf(&handle, key).map_err(|e| StorageError::Io {
            operation: "get",
            message: format!("read failed in CF '{}': {}", cf, e),
            source: Some(Box::new(e)),
        })
    }

    fn delete(&self, cf: ColumnFamilyName, key: &[u8]) -> Result<()> {
        let handle = self.cf_handle(cf)?;
        self.db
            .delete_cf(&handle, key)
            .map_err(|e| StorageError::Io {
                operation: "delete",
                message: format!("delete failed in CF '{}': {}", cf, e),
                source: Some(Box::new(e)),
            })
    }

    fn write_batch(&self, operations: &[BatchOperation]) -> Result<()> {
        let mut batch = WriteBatchWithTransaction::<false>::default();

        for op in operations {
            match op {
                BatchOperation::Put { cf, key, value } => {
                    let handle = self.cf_handle(*cf)?;
                    batch.put_cf(&handle, key, value);
                }
                BatchOperation::Delete { cf, key } => {
                    let handle = self.cf_handle(*cf)?;
                    batch.delete_cf(&handle, key);
                }
            }
        }

        self.db.write(batch).map_err(|e| StorageError::Io {
            operation: "write_batch",
            message: format!("atomic batch write failed: {}", e),
            source: Some(Box::new(e)),
        })
    }

    fn prefix_iterator(
        &self,
        cf: ColumnFamilyName,
        prefix: &[u8],
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let handle = self.cf_handle(cf)?;
        let mut read_opts = rocksdb::ReadOptions::default();
        // Set iterate upper bound to the prefix successor for efficiency.
        // This tells RocksDB to stop iterating once it passes the prefix range.
        if let Some(upper) = prefix_successor(prefix) {
            read_opts.set_iterate_upper_bound(upper);
        }

        let iter = self.db.iterator_cf_opt(
            &handle,
            read_opts,
            rocksdb::IteratorMode::From(prefix, rocksdb::Direction::Forward),
        );

        let mut results = Vec::new();
        for item in iter {
            let (key, value) = item.map_err(|e| StorageError::Io {
                operation: "prefix_iterator",
                message: format!("iteration failed in CF '{}': {}", cf, e),
                source: Some(Box::new(e)),
            })?;
            if !key.starts_with(prefix) {
                break;
            }
            results.push((key.to_vec(), value.to_vec()));
        }
        Ok(results)
    }

    fn range_iterator(
        &self,
        cf: ColumnFamilyName,
        start: &[u8],
        end: &[u8],
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let handle = self.cf_handle(cf)?;
        let mut read_opts = rocksdb::ReadOptions::default();
        read_opts.set_iterate_upper_bound(end.to_vec());

        let iter = self.db.iterator_cf_opt(
            &handle,
            read_opts,
            rocksdb::IteratorMode::From(start, rocksdb::Direction::Forward),
        );

        let mut results = Vec::new();
        for item in iter {
            let (key, value) = item.map_err(|e| StorageError::Io {
                operation: "range_iterator",
                message: format!("iteration failed in CF '{}': {}", cf, e),
                source: Some(Box::new(e)),
            })?;
            results.push((key.to_vec(), value.to_vec()));
        }
        Ok(results)
    }

    fn compact(&self, cf: ColumnFamilyName) -> Result<()> {
        let handle = self.cf_handle(cf)?;
        self.db
            .compact_range_cf(&handle, None::<&[u8]>, None::<&[u8]>);
        Ok(())
    }
}

/// Compute the lexicographic successor of a byte prefix.
/// Returns `None` if the prefix is all `0xFF` bytes (no successor exists).
///
/// Used to set `iterate_upper_bound` for efficient prefix scans —
/// RocksDB can skip entire SST blocks that fall beyond the bound.
fn prefix_successor(prefix: &[u8]) -> Option<Vec<u8>> {
    let mut successor = prefix.to_vec();
    while let Some(last) = successor.last_mut() {
        if *last < 0xFF {
            *last += 1;
            return Some(successor);
        }
        successor.pop();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_backend() -> (RocksDbBackend, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let backend = RocksDbBackend::open(dir.path()).unwrap();
        (backend, dir)
    }

    #[test]
    fn open_creates_all_column_families() {
        let (backend, _dir) = temp_backend();
        for cf in ColumnFamilyName::all() {
            assert!(
                backend.db.cf_handle(cf.as_str()).is_some(),
                "column family '{}' should exist",
                cf
            );
        }
    }

    #[test]
    fn put_get_roundtrip() {
        let (backend, _dir) = temp_backend();
        backend
            .put(ColumnFamilyName::Default, b"key1", b"value1")
            .unwrap();
        let val = backend.get(ColumnFamilyName::Default, b"key1").unwrap();
        assert_eq!(val, Some(b"value1".to_vec()));
    }

    #[test]
    fn get_missing_returns_none() {
        let (backend, _dir) = temp_backend();
        let val = backend
            .get(ColumnFamilyName::Default, b"nonexistent")
            .unwrap();
        assert_eq!(val, None);
    }

    #[test]
    fn delete_removes_key() {
        let (backend, _dir) = temp_backend();
        backend
            .put(ColumnFamilyName::Default, b"key1", b"value1")
            .unwrap();
        backend.delete(ColumnFamilyName::Default, b"key1").unwrap();
        assert_eq!(
            backend.get(ColumnFamilyName::Default, b"key1").unwrap(),
            None
        );
    }

    #[test]
    fn write_batch_atomic() {
        let (backend, _dir) = temp_backend();
        let ops = vec![
            BatchOperation::Put {
                cf: ColumnFamilyName::Default,
                key: b"a".to_vec(),
                value: b"1".to_vec(),
            },
            BatchOperation::Put {
                cf: ColumnFamilyName::Meta,
                key: b"schema_version".to_vec(),
                value: b"1".to_vec(),
            },
        ];
        backend.write_batch(&ops).unwrap();
        assert_eq!(
            backend.get(ColumnFamilyName::Default, b"a").unwrap(),
            Some(b"1".to_vec())
        );
        assert_eq!(
            backend
                .get(ColumnFamilyName::Meta, b"schema_version")
                .unwrap(),
            Some(b"1".to_vec())
        );
    }

    #[test]
    fn prefix_iterator_sorted_and_bounded() {
        let (backend, _dir) = temp_backend();
        backend
            .put(ColumnFamilyName::Default, b"abc_1", b"v1")
            .unwrap();
        backend
            .put(ColumnFamilyName::Default, b"abc_2", b"v2")
            .unwrap();
        backend
            .put(ColumnFamilyName::Default, b"abd_1", b"v3")
            .unwrap();

        let results = backend
            .prefix_iterator(ColumnFamilyName::Default, b"abc")
            .unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, b"abc_1");
        assert_eq!(results[1].0, b"abc_2");
    }

    #[test]
    fn range_iterator_half_open() {
        let (backend, _dir) = temp_backend();
        for i in 0u8..10 {
            backend
                .put(ColumnFamilyName::Default, &[i], &[i * 10])
                .unwrap();
        }
        let results = backend
            .range_iterator(ColumnFamilyName::Default, &[3], &[7])
            .unwrap();
        assert_eq!(results.len(), 4);
        assert_eq!(results[0].0, vec![3]);
        assert_eq!(results[3].0, vec![6]);
    }

    #[test]
    fn column_families_isolated() {
        let (backend, _dir) = temp_backend();
        backend
            .put(ColumnFamilyName::Default, b"key", b"default_val")
            .unwrap();
        backend
            .put(ColumnFamilyName::Meta, b"key", b"meta_val")
            .unwrap();
        assert_eq!(
            backend.get(ColumnFamilyName::Default, b"key").unwrap(),
            Some(b"default_val".to_vec())
        );
        assert_eq!(
            backend.get(ColumnFamilyName::Meta, b"key").unwrap(),
            Some(b"meta_val".to_vec())
        );
        assert_eq!(
            backend.get(ColumnFamilyName::Temporal, b"key").unwrap(),
            None
        );
    }

    #[test]
    fn prefix_successor_works() {
        assert_eq!(prefix_successor(b"abc"), Some(b"abd".to_vec()));
        assert_eq!(prefix_successor(b"ab\xff"), Some(b"ac".to_vec()));
        assert_eq!(prefix_successor(b"\xff\xff"), None);
        assert_eq!(prefix_successor(b""), None);
    }

    #[test]
    fn reopen_preserves_data() {
        let dir = tempfile::tempdir().unwrap();
        {
            let backend = RocksDbBackend::open(dir.path()).unwrap();
            backend
                .put(ColumnFamilyName::Default, b"persist", b"value")
                .unwrap();
        }
        // Re-open after drop
        let backend = RocksDbBackend::open(dir.path()).unwrap();
        assert_eq!(
            backend.get(ColumnFamilyName::Default, b"persist").unwrap(),
            Some(b"value".to_vec())
        );
    }
}
