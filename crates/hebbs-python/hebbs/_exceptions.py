"""HEBBS exception hierarchy.

All HEBBS exceptions inherit from HebbsError. Each subclass carries
structured attributes beyond the message string for programmatic
error handling in framework adapters.
"""

from __future__ import annotations


class HebbsError(Exception):
    """Base exception for all HEBBS operations."""
    pass


class ConnectionError(HebbsError):
    """Server unreachable, DNS failure, or TLS handshake failure."""

    def __init__(self, endpoint: str, reason: str) -> None:
        self.endpoint = endpoint
        self.reason = reason
        super().__init__(f"cannot connect to {endpoint}: {reason}")


class TimeoutError(HebbsError):
    """Operation exceeded the configured deadline."""

    def __init__(self, operation: str, elapsed_seconds: float) -> None:
        self.operation = operation
        self.elapsed_seconds = elapsed_seconds
        super().__init__(f"{operation} timed out after {elapsed_seconds:.1f}s")


class MemoryNotFoundError(HebbsError):
    """Memory ID does not exist."""

    def __init__(self, memory_id: str) -> None:
        self.memory_id = memory_id
        super().__init__(f"memory not found: {memory_id}")


class InvalidInputError(HebbsError):
    """Bad arguments provided to an operation."""

    def __init__(self, field: str, constraint: str, message: str | None = None) -> None:
        self.field = field
        self.constraint = constraint
        msg = message or f"invalid {field}: {constraint}"
        super().__init__(msg)


class StorageError(HebbsError):
    """RocksDB I/O failure (embedded mode)."""
    pass


class EmbeddingError(HebbsError):
    """ONNX model failure (embedded mode)."""
    pass


class ServerError(HebbsError):
    """Server-side bug or internal error."""
    pass


class RateLimitedError(HebbsError):
    """Rate limit exceeded."""

    def __init__(self, retry_after_seconds: float | None = None) -> None:
        self.retry_after_seconds = retry_after_seconds
        msg = "rate limited"
        if retry_after_seconds is not None:
            msg += f", retry after {retry_after_seconds:.1f}s"
        super().__init__(msg)


class SubscriptionClosedError(HebbsError):
    """Subscribe stream ended unexpectedly."""

    def __init__(self, reason: str = "") -> None:
        self.reason = reason
        super().__init__(f"subscription closed: {reason}" if reason else "subscription closed")


class ConfigurationError(HebbsError):
    """Invalid open/connect configuration."""
    pass
