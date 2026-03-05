"""HEBBS Python type definitions.

All public types are dataclasses with full type annotations.
Memory IDs are str (26-character ULID). Context is dict[str, Any].
Enums use Python's enum.Enum with UPPER_CASE naming.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class MemoryKind(enum.Enum):
    """Type of memory record."""
    EPISODE = "episode"
    INSIGHT = "insight"
    REVISION = "revision"


class RecallStrategy(enum.Enum):
    """Recall retrieval strategy."""
    SIMILARITY = "similarity"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    ANALOGICAL = "analogical"


class EdgeType(enum.Enum):
    """Graph edge type between memories."""
    CAUSED_BY = "caused_by"
    RELATED_TO = "related_to"
    FOLLOWED_BY = "followed_by"
    REVISED_FROM = "revised_from"
    INSIGHT_FROM = "insight_from"


class ContextMode(enum.Enum):
    """How context is updated during revision."""
    MERGE = "merge"
    REPLACE = "replace"


@dataclass
class Memory:
    """A HEBBS memory record.

    Memory IDs are 26-character ULID strings. Context is a Python dict.
    Timestamps are microseconds since Unix epoch.
    """

    id: str
    content: str
    importance: float
    context: dict[str, Any] = field(default_factory=dict)
    entity_id: str | None = None
    embedding: list[float] | None = None
    created_at: int = 0
    updated_at: int = 0
    last_accessed_at: int = 0
    access_count: int = 0
    decay_score: float = 0.0
    kind: MemoryKind = MemoryKind.EPISODE
    device_id: str | None = None
    logical_clock: int = 0
    embed_ms: float | None = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Memory):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    @classmethod
    def _from_dict(cls, d: dict[str, Any]) -> Memory:
        """Construct a Memory from a dict returned by the native extension."""
        kind_str = d.get("kind", "episode")
        try:
            kind = MemoryKind(kind_str)
        except ValueError:
            kind = MemoryKind.EPISODE

        return cls(
            id=d.get("id", ""),
            content=d.get("content", ""),
            importance=d.get("importance", 0.5),
            context=d.get("context") or {},
            entity_id=d.get("entity_id"),
            embedding=d.get("embedding"),
            created_at=d.get("created_at", 0),
            updated_at=d.get("updated_at", 0),
            last_accessed_at=d.get("last_accessed_at", 0),
            access_count=d.get("access_count", 0),
            decay_score=d.get("decay_score", 0.0),
            kind=kind,
            device_id=d.get("device_id"),
            logical_clock=d.get("logical_clock", 0),
            embed_ms=d.get("embed_ms"),
        )


@dataclass
class StrategyDetail:
    """Per-strategy metadata attached to a recall result."""

    strategy: str
    relevance: float = 0.0
    distance: float | None = None
    timestamp: int | None = None
    rank: int | None = None
    depth: int | None = None
    edge_type: str | None = None
    seed_id: str | None = None
    embedding_similarity: float | None = None
    structural_similarity: float | None = None

    @classmethod
    def _from_dict(cls, d: dict[str, Any]) -> StrategyDetail:
        return cls(
            strategy=d.get("strategy", ""),
            relevance=d.get("relevance", 0.0),
            distance=d.get("distance"),
            timestamp=d.get("timestamp"),
            rank=d.get("rank"),
            depth=d.get("depth"),
            edge_type=d.get("edge_type"),
            seed_id=d.get("seed_id"),
            embedding_similarity=d.get("embedding_similarity"),
            structural_similarity=d.get("structural_similarity"),
        )


@dataclass
class RecallResult:
    """A single recall result with scored memory."""

    memory: Memory
    score: float
    strategy_details: list[StrategyDetail] = field(default_factory=list)

    @classmethod
    def _from_dict(cls, d: dict[str, Any]) -> RecallResult:
        memory = Memory._from_dict(d.get("memory", {}))
        details = [
            StrategyDetail._from_dict(sd)
            for sd in d.get("strategy_details", [])
        ]
        return cls(memory=memory, score=d.get("score", 0.0), strategy_details=details)


@dataclass
class StrategyError:
    """Error from a specific recall strategy (non-fatal in multi-strategy mode)."""

    strategy: str
    message: str


@dataclass
class RecallOutput:
    """Output from recall()."""

    results: list[RecallResult] = field(default_factory=list)
    strategy_errors: list[StrategyError] = field(default_factory=list)
    embed_ms: float | None = None

    @classmethod
    def _from_dict(cls, d: dict[str, Any]) -> RecallOutput:
        results = [RecallResult._from_dict(r) for r in d.get("results", [])]
        errors = [
            StrategyError(strategy=e.get("strategy", ""), message=e.get("message", ""))
            for e in d.get("strategy_errors", [])
        ]
        return cls(results=results, strategy_errors=errors, embed_ms=d.get("embed_ms"))


@dataclass
class ForgetOutput:
    """Output from forget()."""

    forgotten_count: int = 0
    cascade_count: int = 0
    truncated: bool = False
    tombstone_count: int = 0

    @classmethod
    def _from_dict(cls, d: dict[str, Any]) -> ForgetOutput:
        return cls(
            forgotten_count=d.get("forgotten_count", 0),
            cascade_count=d.get("cascade_count", 0),
            truncated=d.get("truncated", False),
            tombstone_count=d.get("tombstone_count", 0),
        )


@dataclass
class PrimeOutput:
    """Output from prime()."""

    results: list[RecallResult] = field(default_factory=list)
    temporal_count: int = 0
    similarity_count: int = 0

    @classmethod
    def _from_dict(cls, d: dict[str, Any]) -> PrimeOutput:
        results = [RecallResult._from_dict(r) for r in d.get("results", [])]
        return cls(
            results=results,
            temporal_count=d.get("temporal_count", 0),
            similarity_count=d.get("similarity_count", 0),
        )


@dataclass
class ReflectOutput:
    """Output from reflect()."""

    insights_created: int = 0
    clusters_found: int = 0
    clusters_processed: int = 0
    memories_processed: int = 0

    @classmethod
    def _from_dict(cls, d: dict[str, Any]) -> ReflectOutput:
        return cls(
            insights_created=d.get("insights_created", 0),
            clusters_found=d.get("clusters_found", 0),
            clusters_processed=d.get("clusters_processed", 0),
            memories_processed=d.get("memories_processed", 0),
        )


@dataclass
class SubscribePush:
    """A push received from a subscription stream."""

    memory: Memory
    confidence: float
    push_timestamp_us: int = 0

    @classmethod
    def _from_dict(cls, d: dict[str, Any]) -> SubscribePush:
        memory = Memory._from_dict(d.get("memory", {}))
        return cls(
            memory=memory,
            confidence=d.get("confidence", 0.0),
            push_timestamp_us=d.get("push_timestamp_us", 0),
        )


@dataclass
class HealthStatus:
    """Server health status."""

    status: str
    version: str = ""
    memory_count: int = 0
    uptime_seconds: int = 0
