# HEBBS Integration Tests

System-level integration tests live in `crates/hebbs-core/tests/system_tests.rs`
because Cargo workspace integration tests must be part of a crate.

`hebbs-core` is the natural home since it depends on all other workspace crates
(hebbs-storage, hebbs-embed, hebbs-index, hebbs-reflect) and exposes the `Engine`
which is the primary integration surface.

## Running

```bash
# All system tests (excludes scale tests marked #[ignore])
cargo test -p hebbs-core --test system_tests

# Include scale tests (1K, 10K)
cargo test -p hebbs-core --test system_tests -- --ignored

# Single test
cargo test -p hebbs-core --test system_tests test_full_lifecycle
```
