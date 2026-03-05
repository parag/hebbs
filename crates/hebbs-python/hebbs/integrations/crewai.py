"""CrewAI integration for HEBBS.

Provides HebbsShortTermMemory and HebbsLongTermMemory adapters.

Install: pip install hebbs[crewai]
"""

from __future__ import annotations

from typing import Any


class HebbsShortTermMemory:
    """CrewAI short-term memory adapter backed by HEBBS.

    Uses temporal recall with a recent time window.

    Usage:
        from hebbs import HEBBS
        from hebbs.integrations.crewai import HebbsShortTermMemory

        h = HEBBS.open("./agent-memory")
        memory = HebbsShortTermMemory(hebbs=h)
    """

    def __init__(self, hebbs: Any, entity_id: str = "", top_k: int = 10) -> None:
        self.hebbs = hebbs
        self.entity_id = entity_id
        self.top_k = top_k

    def save(self, value: str, metadata: dict[str, Any] | None = None) -> None:
        self.hebbs.remember(
            value,
            importance=0.5,
            context=metadata,
            entity_id=self.entity_id or None,
        )

    def search(self, query: str) -> list[dict[str, Any]]:
        output = self.hebbs.recall(
            query,
            strategy="similarity",
            top_k=self.top_k,
            entity_id=self.entity_id or None,
        )
        return [
            {"content": r.memory.content, "score": r.score, "metadata": r.memory.context}
            for r in output.results
        ]

    def reset(self) -> None:
        if self.entity_id:
            self.hebbs.forget(entity_id=self.entity_id)


class HebbsLongTermMemory:
    """CrewAI long-term memory adapter backed by HEBBS.

    Uses similarity recall across all time.
    """

    def __init__(self, hebbs: Any, entity_id: str = "", top_k: int = 10) -> None:
        self.hebbs = hebbs
        self.entity_id = entity_id
        self.top_k = top_k

    def save(self, value: str, metadata: dict[str, Any] | None = None) -> None:
        self.hebbs.remember(
            value,
            importance=0.7,
            context=metadata,
            entity_id=self.entity_id or None,
        )

    def search(self, query: str) -> list[dict[str, Any]]:
        output = self.hebbs.recall(
            query,
            strategy="similarity",
            top_k=self.top_k,
            entity_id=self.entity_id or None,
        )
        return [
            {"content": r.memory.content, "score": r.score, "metadata": r.memory.context}
            for r in output.results
        ]

    def reset(self) -> None:
        if self.entity_id:
            self.hebbs.forget(entity_id=self.entity_id)
