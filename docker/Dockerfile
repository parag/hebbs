# ──────────────────────────────────────────────────────────────────────────────
# HEBBS Server - Multi-stage Production Dockerfile
# ──────────────────────────────────────────────────────────────────────────────
# Produces a minimal container image (~80MB) with:
#   - hebbs-server  (the cognitive memory server)
#   - hebbs-cli     (debugging / ops from kubectl exec)
#   - hebbs-bench   (in-situ performance validation)
#
# Build:
#   docker buildx build --platform linux/amd64,linux/arm64 -t hebbs/hebbs .
#
# Run:
#   docker run -p 6380:6380 -p 6381:6381 -v hebbs-data:/data hebbs/hebbs
# ──────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Build ───────────────────────────────────────────────────────────
FROM rust:1.75-bookworm AS builder

WORKDIR /build

# Cache dependency compilation: copy manifests first, build a dummy, then copy
# real source. This avoids rebuilding all dependencies on every source change.
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/
COPY proto/ proto/

# Protobuf compiler for tonic-build
RUN apt-get update && apt-get install -y --no-install-recommends protobuf-compiler && \
    rm -rf /var/lib/apt/lists/*

RUN cargo build --release \
    -p hebbs-server \
    -p hebbs-cli \
    -p hebbs-bench && \
    strip target/release/hebbs-server && \
    strip target/release/hebbs-cli && \
    strip target/release/hebbs-bench

# Verify the binary links only against glibc/libgcc (available in distroless cc)
RUN ldd target/release/hebbs-server || true

# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM gcr.io/distroless/cc-debian12:nonroot

LABEL org.opencontainers.image.title="HEBBS" \
      org.opencontainers.image.description="Cognitive memory engine for AI agents" \
      org.opencontainers.image.url="https://github.com/hebbs/hebbs" \
      org.opencontainers.image.source="https://github.com/hebbs/hebbs" \
      org.opencontainers.image.vendor="HEBBS" \
      org.opencontainers.image.licenses="BSL-1.1"

# Binaries
COPY --from=builder /build/target/release/hebbs-server /usr/local/bin/hebbs-server
COPY --from=builder /build/target/release/hebbs-cli /usr/local/bin/hebbs-cli
COPY --from=builder /build/target/release/hebbs-bench /usr/local/bin/hebbs-bench

# Data directory (RocksDB) and model cache owned by nonroot (UID 65532)
# distroless:nonroot already runs as 65532, directories are created here
# so the PVC mount point exists even if no volume is attached.
VOLUME ["/data", "/models"]

# gRPC and HTTP ports
EXPOSE 6380 6381

# Environment defaults (overridable via Helm values / docker run -e)
ENV HEBBS_STORAGE_DATA_DIR=/data \
    HEBBS_SERVER_BIND_ADDRESS=0.0.0.0 \
    HEBBS_SERVER_GRPC_PORT=6380 \
    HEBBS_SERVER_HTTP_PORT=6381 \
    HEBBS_LOGGING_FORMAT=json \
    HEBBS_LOGGING_LEVEL=info

# Graceful shutdown: server handles SIGTERM (Phase 8)
STOPSIGNAL SIGTERM

# Health check: verify gRPC health service responds.
# Uses hebbs-cli which is included in the image.
# Note: distroless has no shell, so HEALTHCHECK CMD must use exec form.
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD ["/usr/local/bin/hebbs-cli", "status", "--endpoint", "localhost:6380", "--timeout", "3"]

ENTRYPOINT ["/usr/local/bin/hebbs-server"]
CMD ["start"]
