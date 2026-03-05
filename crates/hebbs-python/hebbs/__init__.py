"""HEBBS - Cognitive Memory Engine for AI Agents.

Two modes of operation:

    Server mode (connects to a running hebbs-server):
        h = HEBBS("localhost:6380")

    Embedded mode (in-process engine, no server needed):
        h = HEBBS.open("./agent-memory")

All 9 operations are available in both modes with identical signatures.
"""

from __future__ import annotations

from typing import Any, Iterator

from hebbs._exceptions import (
    ConfigurationError,
    ConnectionError,
    EmbeddingError,
    HebbsError,
    InvalidInputError,
    MemoryNotFoundError,
    RateLimitedError,
    ServerError,
    StorageError,
    SubscriptionClosedError,
    TimeoutError,
)
from hebbs._types import (
    ContextMode,
    EdgeType,
    ForgetOutput,
    HealthStatus,
    Memory,
    MemoryKind,
    PrimeOutput,
    RecallOutput,
    RecallResult,
    RecallStrategy,
    ReflectOutput,
    StrategyDetail,
    StrategyError,
    SubscribePush,
)


class SubscribeStream:
    """Wraps a backend subscription with iterator and context manager support."""

    def __init__(self, backend_stream: Any) -> None:
        self._stream = backend_stream

    def feed(self, text: str) -> None:
        """Feed text to the subscription for matching."""
        self._stream.feed(text)

    def poll(self, timeout_secs: float | None = None) -> SubscribePush | None:
        """Poll for the next push. Returns None if no push is available."""
        return self._stream.poll(timeout_secs)

    def close(self) -> None:
        """Close the subscription and release resources."""
        self._stream.close()

    def __enter__(self) -> SubscribeStream:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __iter__(self) -> Iterator[SubscribePush]:
        return self._stream.__iter__()

    def __next__(self) -> SubscribePush:
        return self._stream.__next__()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class HEBBS:
    """HEBBS cognitive memory engine client.

    Server mode:
        h = HEBBS("localhost:6380")

    Embedded mode:
        h = HEBBS.open("./agent-memory")
    """

    def __init__(self, endpoint: str = "localhost:6380", *, timeout: float = 30.0) -> None:
        """Connect to a HEBBS server via gRPC.

        Args:
            endpoint: Server address (host:port).
            timeout: Per-operation timeout in seconds.
        """
        raise ConfigurationError(
            "Server mode (gRPC) requires grpcio. "
            "Install with: pip install hebbs. "
            "For embedded mode, use HEBBS.open('./path')."
        )

    @classmethod
    def open(
        cls,
        data_dir: str,
        *,
        use_mock_embedder: bool = True,
        embedding_dimensions: int = 384,
    ) -> HEBBS:
        """Open an embedded HEBBS engine (no server needed).

        Args:
            data_dir: Path to the database directory.
            use_mock_embedder: Use deterministic mock embedder (True for testing).
            embedding_dimensions: Vector dimensions (default 384).

        Returns:
            HEBBS instance with embedded engine.
        """
        from hebbs._native import NativeBackend

        instance = object.__new__(cls)
        instance._backend = NativeBackend(
            data_dir=data_dir,
            use_mock_embedder=use_mock_embedder,
            embedding_dimensions=embedding_dimensions,
        )
        instance._mode = "embedded"
        return instance

    def remember(
        self,
        content: str,
        *,
        importance: float = 0.5,
        context: dict[str, Any] | None = None,
        entity_id: str | None = None,
    ) -> Memory:
        """Store a new memory.

        Args:
            content: The memory content text.
            importance: Importance score in [0.0, 1.0].
            context: Arbitrary metadata dict.
            entity_id: Optional entity scope.

        Returns:
            The created Memory record.
        """
        return self._backend.remember(
            content=content,
            importance=importance,
            context=context,
            entity_id=entity_id,
        )

    def get(self, memory_id: str) -> Memory:
        """Retrieve a memory by its ULID string.

        Args:
            memory_id: 26-character ULID string.

        Returns:
            The Memory record.

        Raises:
            MemoryNotFoundError: If the memory does not exist.
            InvalidInputError: If the ID is not a valid ULID.
        """
        return self._backend.get(memory_id)

    def recall(
        self,
        cue: str,
        *,
        strategy: RecallStrategy | str = RecallStrategy.SIMILARITY,
        strategies: list[RecallStrategy | str] | None = None,
        top_k: int = 10,
        entity_id: str | None = None,
        max_depth: int | None = None,
        time_range: tuple[int, int] | None = None,
    ) -> RecallOutput:
        """Recall memories matching a cue.

        Args:
            cue: Query text.
            strategy: Single retrieval strategy (used when strategies is None).
            strategies: Multiple strategies to run in parallel with a single
                embedding pass and engine-side merge/dedup. Overrides strategy.
            top_k: Maximum results.
            entity_id: Required for temporal strategy.
            max_depth: Maximum graph traversal depth (causal strategy).
            time_range: (start_us, end_us) for temporal strategy.

        Returns:
            RecallOutput with results and any strategy errors.
        """
        strat_str = strategy.value if isinstance(strategy, RecallStrategy) else str(strategy)

        strat_list: list[str] | None = None
        if strategies is not None:
            strat_list = [
                s.value if isinstance(s, RecallStrategy) else str(s)
                for s in strategies
            ]

        return self._backend.recall(
            cue=cue,
            strategy=strat_str,
            top_k=top_k,
            entity_id=entity_id,
            max_depth=max_depth,
            time_range=time_range,
            strategies=strat_list,
        )

    def revise(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        importance: float | None = None,
        context: dict[str, Any] | None = None,
        context_mode: ContextMode | str = ContextMode.MERGE,
        entity_id: str | None = None,
    ) -> Memory:
        """Revise an existing memory.

        Args:
            memory_id: ULID of the memory to revise.
            content: New content (triggers re-embedding).
            importance: New importance score.
            context: Context updates.
            context_mode: "merge" or "replace".
            entity_id: New entity scope.

        Returns:
            The updated Memory record.
        """
        mode_str = context_mode.value if isinstance(context_mode, ContextMode) else str(context_mode)
        return self._backend.revise(
            memory_id=memory_id,
            content=content,
            importance=importance,
            context=context,
            context_mode=mode_str,
            entity_id=entity_id,
        )

    def forget(
        self,
        memory_id: str | None = None,
        *,
        memory_ids: list[str] | None = None,
        entity_id: str | None = None,
        staleness_threshold_us: int | None = None,
        access_count_floor: int | None = None,
        memory_kind: MemoryKind | str | None = None,
        decay_score_floor: float | None = None,
    ) -> ForgetOutput:
        """Forget memories by ID or criteria.

        Args:
            memory_id: Single memory ULID to forget.
            memory_ids: List of memory ULIDs to forget.
            entity_id: Forget all memories for this entity.
            staleness_threshold_us: Forget memories last accessed before this time.
            access_count_floor: Forget memories with access count below this.
            memory_kind: Forget memories of this kind.
            decay_score_floor: Forget memories with decay score below this.

        Returns:
            ForgetOutput with counts.
        """
        ids = memory_ids or []
        if memory_id is not None:
            ids = [memory_id] + ids

        kind_str = None
        if memory_kind is not None:
            kind_str = memory_kind.value if isinstance(memory_kind, MemoryKind) else str(memory_kind)

        return self._backend.forget(
            memory_ids=ids if ids else None,
            entity_id=entity_id,
            staleness_threshold_us=staleness_threshold_us,
            access_count_floor=access_count_floor,
            memory_kind=kind_str,
            decay_score_floor=decay_score_floor,
        )

    def prime(
        self,
        entity_id: str,
        *,
        max_memories: int = 50,
        similarity_cue: str | None = None,
    ) -> PrimeOutput:
        """Pre-load relevant memories for an entity.

        Args:
            entity_id: Entity to prime for.
            max_memories: Maximum memories to return.
            similarity_cue: Optional text cue for similarity matching.

        Returns:
            PrimeOutput with temporal and similarity results.
        """
        return self._backend.prime(
            entity_id=entity_id,
            max_memories=max_memories,
            similarity_cue=similarity_cue,
        )

    def subscribe(
        self,
        *,
        entity_id: str | None = None,
        confidence_threshold: float = 0.6,
    ) -> SubscribeStream:
        """Start a subscription for real-time memory matching.

        Args:
            entity_id: Optional entity scope.
            confidence_threshold: Minimum confidence to push.

        Returns:
            SubscribeStream with feed(), poll(), and iterator support.
        """
        stream = self._backend.subscribe(
            entity_id=entity_id,
            confidence_threshold=confidence_threshold,
        )
        return SubscribeStream(stream)

    def reflect(
        self,
        *,
        entity_id: str | None = None,
        since_us: int | None = None,
    ) -> ReflectOutput:
        """Trigger reflection to generate insights.

        Args:
            entity_id: Scope reflection to an entity.
            since_us: Only process memories after this timestamp.

        Returns:
            ReflectOutput with insight and cluster counts.
        """
        return self._backend.reflect(
            entity_id=entity_id,
            since_us=since_us,
        )

    def insights(
        self,
        *,
        entity_id: str | None = None,
        min_confidence: float | None = None,
        max_results: int | None = None,
    ) -> list[Memory]:
        """Query stored insights.

        Args:
            entity_id: Filter by entity.
            min_confidence: Minimum confidence threshold.
            max_results: Maximum insights to return.

        Returns:
            List of Memory records of kind INSIGHT.
        """
        return self._backend.insights(
            entity_id=entity_id,
            min_confidence=min_confidence,
            max_results=max_results,
        )

    def count(self) -> int:
        """Get the number of memories in the engine."""
        return self._backend.count()

    def close(self) -> None:
        """Close the engine and release resources."""
        if hasattr(self, "_backend"):
            self._backend.close()

    def __enter__(self) -> HEBBS:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        mode = getattr(self, "_mode", "unknown")
        return f"<HEBBS mode={mode}>"


__all__ = [
    "HEBBS",
    "SubscribeStream",
    "Memory",
    "MemoryKind",
    "RecallStrategy",
    "RecallOutput",
    "RecallResult",
    "StrategyDetail",
    "StrategyError",
    "ForgetOutput",
    "PrimeOutput",
    "ReflectOutput",
    "SubscribePush",
    "HealthStatus",
    "EdgeType",
    "ContextMode",
    "HebbsError",
    "ConnectionError",
    "TimeoutError",
    "MemoryNotFoundError",
    "InvalidInputError",
    "StorageError",
    "EmbeddingError",
    "ServerError",
    "RateLimitedError",
    "SubscriptionClosedError",
    "ConfigurationError",
]
