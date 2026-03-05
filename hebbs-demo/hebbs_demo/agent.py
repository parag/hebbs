"""SalesAgent: the core conversation loop with full HEBBS integration.

Ties together:
  - LLM conversation generation
  - Memory extraction and storage
  - Multi-strategy recall for context building
  - Subscribe for real-time memory surfacing
  - Prime for session initialization
  - Reflect for institutional learning
  - Display manager for observability
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from hebbs_demo.config import DemoConfig
from hebbs_demo.display import DisplayManager, OperationRecord, TimedOperation, Verbosity
from hebbs_demo.llm_client import LlmClient, MockLlmClient
from hebbs_demo.memory_manager import MemoryManager
from hebbs_demo.prompts import conversation_prompt

logger = logging.getLogger(__name__)


@dataclass
class TurnResult:
    """Result of a single conversation turn."""
    prospect_message: str
    agent_response: str
    memories_created: int
    memories_recalled: int
    subscribe_pushes: int
    turn_latency_ms: float


@dataclass
class SessionResult:
    """Result of a complete conversation session."""
    entity_id: str
    turns: list[TurnResult] = field(default_factory=list)
    primed_memories: int = 0
    total_memories_created: int = 0
    total_memories_recalled: int = 0
    total_subscribe_pushes: int = 0


class SalesAgent:
    """AI Sales Intelligence Agent powered by HEBBS."""

    def __init__(
        self,
        config: DemoConfig,
        hebbs: Any,
        llm_client: LlmClient | None = None,
        display: DisplayManager | None = None,
        use_mock_llm: bool = False,
    ) -> None:
        self._config = config
        self._hebbs = hebbs
        self._display = display or DisplayManager()

        if use_mock_llm:
            self._llm = MockLlmClient(config)
        else:
            self._llm = llm_client or LlmClient(config)

        self._memory_mgr = MemoryManager(hebbs, self._llm, self._display)
        self._session_history: list[dict[str, str]] = []
        self._current_entity: str | None = None
        self._subscription: Any = None

    @property
    def llm_client(self) -> LlmClient:
        return self._llm

    @property
    def memory_manager(self) -> MemoryManager:
        return self._memory_mgr

    @property
    def hebbs(self) -> Any:
        return self._hebbs

    def start_session(
        self,
        entity_id: str,
        session_num: int | None = None,
        use_subscribe: bool = False,
        similarity_cue: str | None = None,
    ) -> str:
        """Initialize a new conversation session with an entity.

        Returns the primed context string.
        """
        self._current_entity = entity_id
        self._session_history = []
        self._display.display_session_header(entity_id, session_num)

        context, _ = self._memory_mgr.prime_session(
            entity_id=entity_id,
            similarity_cue=similarity_cue,
        )

        if use_subscribe:
            try:
                self._subscription = self._hebbs.subscribe(
                    entity_id=entity_id,
                    confidence_threshold=0.5,
                )
            except Exception as e:
                logger.warning("subscribe() failed: %s", e)
                self._subscription = None

        return context

    def end_session(self) -> None:
        """Clean up the current session."""
        if self._subscription is not None:
            try:
                self._subscription.close()
            except Exception:
                pass
            self._subscription = None
        self._session_history = []
        self._current_entity = None

    def process_turn(
        self,
        prospect_message: str,
        recall_strategies: list[str] | None = None,
    ) -> TurnResult:
        """Process a single conversation turn.

        1. Display prospect message
        2. Feed to subscribe (if active)
        3. Recall relevant memories
        4. Generate agent response via LLM
        5. Extract and store memories from the turn
        6. Return results
        """
        t0 = time.perf_counter()
        self._display.start_turn()
        entity = self._current_entity

        self._display.display_prospect_message(entity or "Prospect", prospect_message)

        # Feed to subscribe for real-time surfacing
        subscribe_pushes: list[Any] = []
        if self._subscription is not None:
            with TimedOperation() as sub_timer:
                try:
                    self._subscription.feed(prospect_message)
                    time.sleep(0.05)  # brief wait for async pipeline
                    while True:
                        push = self._subscription.poll(timeout_secs=0.1)
                        if push is None:
                            break
                        subscribe_pushes.append(push)
                except Exception as e:
                    logger.warning("subscribe feed/poll failed: %s", e)

            if subscribe_pushes:
                details = []
                for p in subscribe_pushes:
                    details.append(
                        f'"{p.memory.content[:55]}" (confidence: {p.confidence:.2f})'
                    )
                record = OperationRecord(
                    operation="SUBSCRIBE",
                    latency_ms=sub_timer.elapsed_ms,
                    summary=f"{len(subscribe_pushes)} memory surfaced (confidence: {subscribe_pushes[0].confidence:.2f})",
                    details=details,
                    highlight_color="yellow",
                )
                self._display.record_operation(record)

        # Recall relevant memories
        strategies = recall_strategies or ["similarity"]
        recalled_context, recall_results = self._memory_mgr.recall_context(
            cue=prospect_message,
            entity_id=entity,
            strategies=strategies,
        )

        # Add subscribe context
        subscribe_context = self._memory_mgr.get_subscribe_context(subscribe_pushes)
        full_context = recalled_context
        if subscribe_context:
            full_context += "\n\n--- REAL-TIME SURFACED ---\n" + subscribe_context

        # Get insights string
        insights_str = ""
        try:
            insights = self._hebbs.insights(entity_id=entity, max_results=5)
            if insights:
                insights_str = "\n".join(f"- {ins.content}" for ins in insights)
        except Exception:
            pass

        messages = conversation_prompt(
            prospect_message=prospect_message,
            recalled_context=full_context,
            session_history=self._session_history,
            entity_id=entity,
            insights=insights_str,
        )

        with TimedOperation() as llm_timer:
            llm_resp = self._llm.conversation(messages)

        agent_response = llm_resp.content

        llm_details = [
            f"model: {llm_resp.model}  |  provider: {llm_resp.provider}",
            f"tokens: {llm_resp.input_tokens} in / {llm_resp.output_tokens} out",
        ]
        self._display.record_operation(OperationRecord(
            operation="LLM CHAT",
            latency_ms=llm_timer.elapsed_ms,
            summary=f"response generated ({llm_resp.output_tokens} tokens)",
            details=llm_details,
            highlight_color="yellow",
            llm_ms=llm_timer.elapsed_ms,
        ))

        # Update session history
        self._session_history.append({"role": "user", "content": prospect_message})
        self._session_history.append({"role": "assistant", "content": agent_response})

        # Extract and store memories
        stored = self._memory_mgr.extract_and_remember(
            prospect_message=prospect_message,
            agent_response=agent_response,
            entity_id=entity,
            recalled_context=recalled_context,
        )

        # Display turn activity and response
        self._display.display_turn()
        self._display.display_agent_response(agent_response)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return TurnResult(
            prospect_message=prospect_message,
            agent_response=agent_response,
            memories_created=len(stored),
            memories_recalled=len(recall_results),
            subscribe_pushes=len(subscribe_pushes),
            turn_latency_ms=elapsed_ms,
        )

    def run_reflect(self, entity_id: str | None = None) -> Any:
        """Trigger the reflect pipeline and display results."""
        with TimedOperation() as timer:
            try:
                result = self._hebbs.reflect(entity_id=entity_id)
            except Exception as e:
                logger.warning("reflect() failed: %s", e)
                return None

        self._display.display_reflect(
            memories_processed=result.memories_processed,
            clusters_found=result.clusters_found,
            insights_created=result.insights_created,
            latency_ms=timer.elapsed_ms,
        )
        return result

    def run_forget(self, entity_id: str) -> Any:
        """Forget all memories for an entity and display results."""
        with TimedOperation() as timer:
            try:
                result = self._hebbs.forget(entity_id=entity_id)
            except Exception as e:
                logger.warning("forget() failed: %s", e)
                return None

        self._display.display_forget(
            entity_id=entity_id,
            forgotten_count=result.forgotten_count,
            cascade_count=result.cascade_count,
            tombstone_count=result.tombstone_count,
            latency_ms=timer.elapsed_ms,
        )
        return result
