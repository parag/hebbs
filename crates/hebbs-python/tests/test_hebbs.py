"""Comprehensive test suite for the HEBBS Python SDK.

Tests cover:
- Type definitions (dataclasses, enums, exceptions)
- Embedded mode full lifecycle (all 9 operations)
- Error handling and edge cases
- Concurrent access from multiple threads
- Context manager protocol
- Subscribe lifecycle
- Reflect and insights pipeline
- Data type round-trips (context dicts, timestamps, floats)
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
from typing import Any

import pytest

import hebbs
from hebbs import (
    HEBBS,
    ContextMode,
    EdgeType,
    ForgetOutput,
    HebbsError,
    InvalidInputError,
    Memory,
    MemoryKind,
    MemoryNotFoundError,
    PrimeOutput,
    RecallOutput,
    RecallResult,
    RecallStrategy,
    ReflectOutput,
    StrategyDetail,
    SubscribePush,
)
from hebbs._exceptions import ConfigurationError, StorageError


# ═══════════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def engine():
    """Create an embedded HEBBS engine in a temporary directory."""
    with tempfile.TemporaryDirectory() as td:
        h = HEBBS.open(td, use_mock_embedder=True, embedding_dimensions=8)
        yield h
        h.close()


@pytest.fixture
def populated_engine():
    """Create an engine pre-populated with 20 memories across 2 entities."""
    with tempfile.TemporaryDirectory() as td:
        h = HEBBS.open(td, use_mock_embedder=True, embedding_dimensions=8)
        for i in range(10):
            h.remember(
                f"Memory about topic A number {i}",
                importance=0.5 + (i * 0.05),
                entity_id="entity_a",
                context={"index": i, "category": "topic_a"},
            )
        for i in range(10):
            h.remember(
                f"Memory about topic B number {i}",
                importance=0.3 + (i * 0.05),
                entity_id="entity_b",
                context={"index": i, "category": "topic_b"},
            )
        yield h
        h.close()


# ═══════════════════════════════════════════════════════════════════════
#  Type tests
# ═══════════════════════════════════════════════════════════════════════


class TestTypes:
    """Test Python type definitions."""

    def test_memory_kind_values(self):
        assert MemoryKind.EPISODE.value == "episode"
        assert MemoryKind.INSIGHT.value == "insight"
        assert MemoryKind.REVISION.value == "revision"

    def test_recall_strategy_values(self):
        assert RecallStrategy.SIMILARITY.value == "similarity"
        assert RecallStrategy.TEMPORAL.value == "temporal"
        assert RecallStrategy.CAUSAL.value == "causal"
        assert RecallStrategy.ANALOGICAL.value == "analogical"

    def test_edge_type_values(self):
        assert EdgeType.CAUSED_BY.value == "caused_by"
        assert EdgeType.RELATED_TO.value == "related_to"
        assert EdgeType.FOLLOWED_BY.value == "followed_by"

    def test_context_mode_values(self):
        assert ContextMode.MERGE.value == "merge"
        assert ContextMode.REPLACE.value == "replace"

    def test_memory_from_dict(self):
        d = {
            "id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
            "content": "test content",
            "importance": 0.8,
            "context": {"key": "value"},
            "entity_id": "ent1",
            "kind": "episode",
            "created_at": 1000000,
            "updated_at": 1000000,
            "last_accessed_at": 1000000,
            "access_count": 5,
            "decay_score": 0.7,
        }
        m = Memory._from_dict(d)
        assert m.id == "01ARZ3NDEKTSV4RRFFQ69G5FAV"
        assert m.content == "test content"
        assert m.importance == 0.8
        assert m.context == {"key": "value"}
        assert m.entity_id == "ent1"
        assert m.kind == MemoryKind.EPISODE
        assert m.access_count == 5

    def test_memory_equality_by_id(self):
        m1 = Memory(id="abc", content="x", importance=0.5)
        m2 = Memory(id="abc", content="y", importance=0.9)
        m3 = Memory(id="def", content="x", importance=0.5)
        assert m1 == m2
        assert m1 != m3

    def test_memory_hash_by_id(self):
        m1 = Memory(id="abc", content="x", importance=0.5)
        m2 = Memory(id="abc", content="y", importance=0.9)
        assert hash(m1) == hash(m2)

    def test_recall_output_from_dict(self):
        d = {
            "results": [
                {
                    "memory": {"id": "test", "content": "c", "importance": 0.5, "kind": "episode"},
                    "score": 0.9,
                    "strategy_details": [
                        {"strategy": "similarity", "distance": 0.1, "relevance": 0.9}
                    ],
                }
            ],
            "strategy_errors": [],
        }
        output = RecallOutput._from_dict(d)
        assert len(output.results) == 1
        assert output.results[0].score == 0.9
        assert output.results[0].memory.content == "c"

    def test_forget_output_from_dict(self):
        d = {"forgotten_count": 3, "cascade_count": 1, "truncated": False, "tombstone_count": 3}
        fo = ForgetOutput._from_dict(d)
        assert fo.forgotten_count == 3
        assert fo.cascade_count == 1
        assert not fo.truncated


# ═══════════════════════════════════════════════════════════════════════
#  Exception tests
# ═══════════════════════════════════════════════════════════════════════


class TestExceptions:
    """Test exception hierarchy and attributes."""

    def test_all_exceptions_inherit_from_hebbs_error(self):
        from hebbs._exceptions import (
            ConnectionError,
            EmbeddingError,
            InvalidInputError,
            MemoryNotFoundError,
            RateLimitedError,
            ServerError,
            StorageError,
            SubscriptionClosedError,
            TimeoutError,
        )

        for exc_cls in [
            ConnectionError, TimeoutError, MemoryNotFoundError,
            InvalidInputError, StorageError, EmbeddingError,
            ServerError, RateLimitedError, SubscriptionClosedError,
        ]:
            assert issubclass(exc_cls, HebbsError)

    def test_memory_not_found_error_attributes(self):
        err = MemoryNotFoundError("01ARZ3NDEKTSV4RRFFQ69G5FAV")
        assert err.memory_id == "01ARZ3NDEKTSV4RRFFQ69G5FAV"
        assert "01ARZ3NDEKTSV4RRFFQ69G5FAV" in str(err)

    def test_invalid_input_error_attributes(self):
        err = InvalidInputError("content", "max 65536 bytes")
        assert err.field == "content"
        assert err.constraint == "max 65536 bytes"

    def test_configuration_error_for_native_unavailable(self):
        assert issubclass(ConfigurationError, HebbsError)


# ═══════════════════════════════════════════════════════════════════════
#  Engine lifecycle tests
# ═══════════════════════════════════════════════════════════════════════


class TestEngineLifecycle:
    """Test engine open/close lifecycle."""

    def test_open_and_close(self):
        with tempfile.TemporaryDirectory() as td:
            h = HEBBS.open(td, use_mock_embedder=True, embedding_dimensions=8)
            assert h.count() == 0
            h.close()

    def test_context_manager(self):
        with tempfile.TemporaryDirectory() as td:
            with HEBBS.open(td, use_mock_embedder=True, embedding_dimensions=8) as h:
                h.remember("test", importance=0.5)
                assert h.count() == 1

    def test_repr(self):
        with tempfile.TemporaryDirectory() as td:
            h = HEBBS.open(td, use_mock_embedder=True, embedding_dimensions=8)
            assert "embedded" in repr(h)
            h.close()

    def test_open_invalid_path_raises(self):
        with pytest.raises(HebbsError):
            HEBBS.open("/nonexistent/path/that/should/fail")


# ═══════════════════════════════════════════════════════════════════════
#  Remember tests
# ═══════════════════════════════════════════════════════════════════════


class TestRemember:
    """Test the remember() operation."""

    def test_remember_basic(self, engine):
        m = engine.remember("hello world", importance=0.5)
        assert isinstance(m, Memory)
        assert m.content == "hello world"
        assert len(m.id) == 26
        assert m.kind == MemoryKind.EPISODE

    def test_remember_with_importance(self, engine):
        m = engine.remember("important thing", importance=0.95)
        assert abs(m.importance - 0.95) < 0.01

    def test_remember_with_context(self, engine):
        ctx = {"stage": "discovery", "priority": 1, "tags": ["sales", "urgent"]}
        m = engine.remember("with context", importance=0.5, context=ctx)
        assert m.context.get("stage") == "discovery"
        assert m.context.get("priority") == 1
        assert m.context.get("tags") == ["sales", "urgent"]

    def test_remember_with_entity_id(self, engine):
        m = engine.remember("scoped", importance=0.5, entity_id="customer_123")
        assert m.entity_id == "customer_123"

    def test_remember_increments_count(self, engine):
        assert engine.count() == 0
        engine.remember("one", importance=0.5)
        assert engine.count() == 1
        engine.remember("two", importance=0.5)
        assert engine.count() == 2

    def test_remember_unique_ids(self, engine):
        ids = set()
        for i in range(100):
            m = engine.remember(f"memory {i}", importance=0.5)
            ids.add(m.id)
        assert len(ids) == 100

    def test_remember_empty_content_raises(self, engine):
        with pytest.raises(HebbsError):
            engine.remember("", importance=0.5)

    def test_remember_timestamps_set(self, engine):
        m = engine.remember("timestamped", importance=0.5)
        assert m.created_at > 0
        assert m.updated_at > 0
        assert m.created_at == m.updated_at


# ═══════════════════════════════════════════════════════════════════════
#  Get tests
# ═══════════════════════════════════════════════════════════════════════


class TestGet:
    """Test the get() operation."""

    def test_get_existing(self, engine):
        m = engine.remember("retrievable", importance=0.7)
        fetched = engine.get(m.id)
        assert fetched.id == m.id
        assert fetched.content == "retrievable"
        assert abs(fetched.importance - 0.7) < 0.01

    def test_get_nonexistent_raises(self, engine):
        with pytest.raises(HebbsError):
            engine.get("01ARZ3NDEKTSV4RRFFQ69G5FAV")

    def test_get_invalid_ulid_raises(self, engine):
        with pytest.raises((HebbsError, ValueError)):
            engine.get("not-a-ulid")

    def test_get_preserves_context(self, engine):
        ctx = {"nested": {"deep": True}, "list": [1, 2, 3]}
        m = engine.remember("ctx test", importance=0.5, context=ctx)
        fetched = engine.get(m.id)
        assert fetched.context.get("nested") == {"deep": True}
        assert fetched.context.get("list") == [1, 2, 3]


# ═══════════════════════════════════════════════════════════════════════
#  Recall tests
# ═══════════════════════════════════════════════════════════════════════


class TestRecall:
    """Test the recall() operation."""

    def test_recall_similarity(self, populated_engine):
        output = populated_engine.recall("topic A information", top_k=5)
        assert isinstance(output, RecallOutput)
        assert len(output.results) <= 5
        for r in output.results:
            assert isinstance(r, RecallResult)
            assert isinstance(r.memory, Memory)
            assert r.score >= 0

    def test_recall_temporal(self, populated_engine):
        output = populated_engine.recall(
            "topic A",
            strategy=RecallStrategy.TEMPORAL,
            entity_id="entity_a",
            top_k=5,
        )
        assert isinstance(output, RecallOutput)

    def test_recall_temporal_string_strategy(self, populated_engine):
        output = populated_engine.recall(
            "topic A",
            strategy="temporal",
            entity_id="entity_a",
            top_k=5,
        )
        assert isinstance(output, RecallOutput)

    def test_recall_empty_database(self, engine):
        output = engine.recall("anything")
        assert len(output.results) == 0

    def test_recall_empty_cue_raises(self, engine):
        with pytest.raises(HebbsError):
            engine.recall("")

    def test_recall_invalid_strategy_raises(self, engine):
        with pytest.raises((HebbsError, ValueError)):
            engine.recall("test", strategy="nonexistent")

    def test_recall_results_have_strategy_details(self, populated_engine):
        output = populated_engine.recall("topic A", top_k=3)
        if output.results:
            assert len(output.results[0].strategy_details) > 0
            assert output.results[0].strategy_details[0].strategy == "similarity"


# ═══════════════════════════════════════════════════════════════════════
#  Revise tests
# ═══════════════════════════════════════════════════════════════════════


class TestRevise:
    """Test the revise() operation."""

    def test_revise_content(self, engine):
        m = engine.remember("original content", importance=0.5)
        revised = engine.revise(m.id, content="updated content")
        assert revised.content == "updated content"
        assert revised.id == m.id

    def test_revise_importance(self, engine):
        m = engine.remember("test", importance=0.3)
        revised = engine.revise(m.id, importance=0.9)
        assert abs(revised.importance - 0.9) < 0.01

    def test_revise_context_merge(self, engine):
        m = engine.remember("test", importance=0.5, context={"a": 1, "b": 2})
        revised = engine.revise(m.id, context={"b": 99, "c": 3}, context_mode="merge")
        ctx = revised.context
        assert ctx.get("a") == 1
        assert ctx.get("b") == 99
        assert ctx.get("c") == 3

    def test_revise_context_replace(self, engine):
        m = engine.remember("test", importance=0.5, context={"a": 1, "b": 2})
        revised = engine.revise(
            m.id, context={"x": 10}, context_mode=ContextMode.REPLACE
        )
        ctx = revised.context
        assert "a" not in ctx
        assert ctx.get("x") == 10

    def test_revise_nonexistent_raises(self, engine):
        with pytest.raises(HebbsError):
            engine.revise("01ARZ3NDEKTSV4RRFFQ69G5FAV", content="nope")

    def test_revise_persists(self, engine):
        m = engine.remember("before", importance=0.5)
        engine.revise(m.id, content="after")
        fetched = engine.get(m.id)
        assert fetched.content == "after"


# ═══════════════════════════════════════════════════════════════════════
#  Forget tests
# ═══════════════════════════════════════════════════════════════════════


class TestForget:
    """Test the forget() operation."""

    def test_forget_by_id(self, engine):
        m = engine.remember("to forget", importance=0.5)
        assert engine.count() == 1
        output = engine.forget(m.id)
        assert isinstance(output, ForgetOutput)
        assert output.forgotten_count == 1

    def test_forget_by_entity(self, populated_engine):
        initial = populated_engine.count()
        output = populated_engine.forget(entity_id="entity_a")
        assert output.forgotten_count == 10
        assert populated_engine.count() == initial - 10

    def test_forget_nonexistent_is_noop(self, engine):
        output = engine.forget("01ARZ3NDEKTSV4RRFFQ69G5FAV")
        assert output.forgotten_count == 0

    def test_forget_removes_from_recall(self, engine):
        m = engine.remember("findable", importance=0.8)
        output = engine.recall("findable")
        assert len(output.results) > 0
        engine.forget(m.id)
        output = engine.recall("findable")
        assert len(output.results) == 0

    def test_forget_multiple_ids(self, engine):
        ids = []
        for i in range(5):
            m = engine.remember(f"mem {i}", importance=0.5)
            ids.append(m.id)
        output = engine.forget(memory_ids=ids)
        assert output.forgotten_count == 5
        assert engine.count() == 0


# ═══════════════════════════════════════════════════════════════════════
#  Prime tests
# ═══════════════════════════════════════════════════════════════════════


class TestPrime:
    """Test the prime() operation."""

    def test_prime_basic(self, populated_engine):
        output = populated_engine.prime("entity_a", max_memories=5)
        assert isinstance(output, PrimeOutput)
        assert len(output.results) <= 5

    def test_prime_empty_entity(self, populated_engine):
        output = populated_engine.prime("nonexistent_entity", max_memories=5)
        assert output.temporal_count == 0

    def test_prime_returns_recall_results(self, populated_engine):
        output = populated_engine.prime("entity_a", max_memories=10)
        for r in output.results:
            assert isinstance(r, RecallResult)
            assert isinstance(r.memory, Memory)


# ═══════════════════════════════════════════════════════════════════════
#  Reflect and Insights tests
# ═══════════════════════════════════════════════════════════════════════


class TestReflect:
    """Test reflect() and insights() operations."""

    def test_reflect_basic(self, populated_engine):
        output = populated_engine.reflect(entity_id="entity_a")
        assert isinstance(output, ReflectOutput)
        assert output.memories_processed >= 0

    def test_reflect_global(self, populated_engine):
        output = populated_engine.reflect()
        assert isinstance(output, ReflectOutput)

    def test_insights_empty(self, engine):
        results = engine.insights()
        assert results == []

    def test_insights_returns_memories(self, populated_engine):
        populated_engine.reflect(entity_id="entity_a")
        results = populated_engine.insights(entity_id="entity_a")
        assert isinstance(results, list)
        for m in results:
            assert isinstance(m, Memory)


# ═══════════════════════════════════════════════════════════════════════
#  Subscribe tests
# ═══════════════════════════════════════════════════════════════════════


class TestSubscribe:
    """Test the subscribe() operation."""

    def test_subscribe_open_close(self, engine):
        engine.remember("background memory", importance=0.5)
        with engine.subscribe(confidence_threshold=0.1) as stream:
            assert stream is not None
            stream.feed("some text about background memory")
            push = stream.poll(timeout_secs=0.5)

    def test_subscribe_context_manager(self, engine):
        engine.remember("test memory", importance=0.5)
        with engine.subscribe() as stream:
            stream.feed("test")
            time.sleep(0.1)

    def test_subscribe_feed_and_poll(self, engine):
        for i in range(5):
            engine.remember(f"indexed memory number {i}", importance=0.8)
        with engine.subscribe(confidence_threshold=0.01) as stream:
            stream.feed("indexed memory information")
            time.sleep(0.5)
            push = stream.poll()


# ═══════════════════════════════════════════════════════════════════════
#  Concurrent access tests
# ═══════════════════════════════════════════════════════════════════════


class TestConcurrency:
    """Test concurrent access from multiple threads."""

    def test_concurrent_remember(self, engine):
        """4 threads each remember 50 memories concurrently."""
        errors = []
        ids_per_thread: list[list[str]] = [[] for _ in range(4)]

        def worker(thread_idx: int):
            try:
                for i in range(50):
                    m = engine.remember(
                        f"thread {thread_idx} memory {i}",
                        importance=0.5,
                        entity_id=f"thread_{thread_idx}",
                    )
                    ids_per_thread[thread_idx].append(m.id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Thread errors: {errors}"
        all_ids = [id for ids in ids_per_thread for id in ids]
        assert len(all_ids) == 200
        assert len(set(all_ids)) == 200

    def test_concurrent_remember_and_recall(self, engine):
        """Concurrent writes and reads do not corrupt data."""
        for i in range(20):
            engine.remember(f"seed memory {i}", importance=0.5)

        errors = []

        def writer():
            try:
                for i in range(30):
                    engine.remember(f"concurrent write {i}", importance=0.5)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(30):
                    engine.recall("memory", top_k=5)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Thread errors: {errors}"


# ═══════════════════════════════════════════════════════════════════════
#  Full lifecycle test
# ═══════════════════════════════════════════════════════════════════════


class TestFullLifecycle:
    """End-to-end lifecycle test."""

    def test_remember_recall_revise_forget(self):
        with tempfile.TemporaryDirectory() as td:
            with HEBBS.open(td, use_mock_embedder=True, embedding_dimensions=8) as h:
                m = h.remember(
                    "customer budget is 100k",
                    importance=0.8,
                    entity_id="deal_42",
                    context={"stage": "negotiation"},
                )
                assert h.count() == 1

                output = h.recall("budget information", top_k=5)
                assert len(output.results) >= 1
                assert output.results[0].memory.id == m.id

                revised = h.revise(
                    m.id,
                    content="customer budget revised to 150k",
                    importance=0.9,
                    context={"stage": "closing"},
                )
                assert revised.content == "customer budget revised to 150k"
                assert abs(revised.importance - 0.9) < 0.01

                fetched = h.get(m.id)
                assert fetched.content == "customer budget revised to 150k"

                forget_out = h.forget(m.id)
                assert forget_out.forgotten_count >= 1

                with pytest.raises(HebbsError):
                    h.get(m.id)


# ═══════════════════════════════════════════════════════════════════════
#  Context round-trip tests
# ═══════════════════════════════════════════════════════════════════════


class TestContextRoundTrip:
    """Test that complex context dicts survive the Python→Rust→Python conversion."""

    def test_nested_dict(self, engine):
        ctx = {"level1": {"level2": {"level3": "deep_value"}}}
        m = engine.remember("nested", importance=0.5, context=ctx)
        fetched = engine.get(m.id)
        assert fetched.context["level1"]["level2"]["level3"] == "deep_value"

    def test_mixed_types(self, engine):
        ctx = {
            "string": "hello",
            "integer": 42,
            "float": 3.14,
            "boolean": True,
            "null": None,
            "list": [1, "two", 3.0],
        }
        m = engine.remember("mixed", importance=0.5, context=ctx)
        fetched = engine.get(m.id)
        assert fetched.context["string"] == "hello"
        assert fetched.context["integer"] == 42
        assert abs(fetched.context["float"] - 3.14) < 0.01
        assert fetched.context["boolean"] is True
        assert fetched.context["null"] is None
        assert fetched.context["list"] == [1, "two", 3.0]

    def test_empty_context(self, engine):
        m = engine.remember("no context", importance=0.5)
        fetched = engine.get(m.id)
        assert fetched.context == {}

    def test_unicode_context(self, engine):
        ctx = {"emoji": "🎉", "chinese": "你好", "arabic": "مرحبا"}
        m = engine.remember("unicode", importance=0.5, context=ctx)
        fetched = engine.get(m.id)
        assert fetched.context["emoji"] == "🎉"
        assert fetched.context["chinese"] == "你好"
        assert fetched.context["arabic"] == "مرحبا"


# ═══════════════════════════════════════════════════════════════════════
#  Scale test
# ═══════════════════════════════════════════════════════════════════════


class TestScale:
    """Test at moderate scale."""

    def test_remember_1000(self):
        with tempfile.TemporaryDirectory() as td:
            with HEBBS.open(td, use_mock_embedder=True, embedding_dimensions=8) as h:
                for i in range(1000):
                    h.remember(f"memory number {i} with some content", importance=0.5)
                assert h.count() == 1000

                output = h.recall("memory content", top_k=10)
                assert len(output.results) == 10

    def test_recall_at_1000(self):
        with tempfile.TemporaryDirectory() as td:
            with HEBBS.open(td, use_mock_embedder=True, embedding_dimensions=8) as h:
                for i in range(1000):
                    h.remember(
                        f"memory about topic {i % 10} instance {i}",
                        importance=0.5,
                        entity_id=f"entity_{i % 5}",
                    )
                output = h.recall("topic 3", top_k=20)
                assert len(output.results) == 20

                output = h.recall(
                    "topic",
                    strategy="temporal",
                    entity_id="entity_0",
                    top_k=50,
                )
                assert isinstance(output, RecallOutput)
