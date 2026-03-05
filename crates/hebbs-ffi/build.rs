fn main() {
    // cbindgen header generation happens via `cargo run --example generate_header`
    // or manually: `cbindgen --config cbindgen.toml --crate hebbs-ffi --output include/hebbs.h`
    //
    // We don't auto-generate in build.rs to avoid requiring cbindgen as a
    // mandatory build dependency. The header is committed to the repository.
}
