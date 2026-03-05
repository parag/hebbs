"""LangChain integration for HEBBS.

Provides HebbsMemory (BaseMemory adapter) and HebbsVectorStore
for use as a LangChain retriever.

Install: pip install hebbs[langchain]
"""

from __future__ import annotations

from typing import Any

try:
    from langchain_core.memory import BaseMemory
except ImportError:
    raise ImportError(
        "langchain-core is required for the LangChain integration. "
        "Install with: pip install hebbs[langchain]"
    )


class HebbsMemory(BaseMemory):
    """LangChain BaseMemory adapter backed by HEBBS.

    Usage:
        from hebbs import HEBBS
        from hebbs.integrations.langchain import HebbsMemory

        h = HEBBS.open("./agent-memory")
        memory = HebbsMemory(hebbs=h, entity_id="conversation_123")
    """

    hebbs: Any
    entity_id: str = ""
    memory_key: str = "history"
    input_key: str = "input"
    output_key: str = "output"
    top_k: int = 5

    @property
    def memory_variables(self) -> list[str]:
        return [self.memory_key]

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        query = inputs.get(self.input_key, "")
        if not query:
            return {self.memory_key: []}
        output = self.hebbs.recall(str(query), top_k=self.top_k, entity_id=self.entity_id or None)
        memories = [
            {"content": r.memory.content, "importance": r.memory.importance, "score": r.score}
            for r in output.results
        ]
        return {self.memory_key: memories}

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        text = outputs.get(self.output_key, "")
        if text:
            self.hebbs.remember(
                str(text),
                importance=0.5,
                entity_id=self.entity_id or None,
            )

    def clear(self) -> None:
        if self.entity_id:
            self.hebbs.forget(entity_id=self.entity_id)
