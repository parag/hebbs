"""Embedded mode backend wrapping the PyO3 native extension.

Imports hebbs._hebbs_native.NativeEngine and wraps it with
Python-native types (dataclasses, enums, exceptions).
"""

from __future__ import annotations

from typing import Any, Iterator

from hebbs._exceptions import (
    ConfigurationError,
    HebbsError,
    InvalidInputError,
    MemoryNotFoundError,
    StorageError,
    EmbeddingError,
)
from hebbs._types import (
    ForgetOutput,
    Memory,
    MemoryKind,
    PrimeOutput,
    RecallOutput,
    RecallStrategy,
    ReflectOutput,
    SubscribePush,
)


def _import_native():
    """Import the native extension, raising a clear error if unavailable."""
    try:
        from hebbs import _hebbs_native
        return _hebbs_native
    except ImportError:
        raise ConfigurationError(
            "Native extension not available. Install a platform-specific wheel "
            "for embedded mode, or use server mode: HEBBS('localhost:6380')"
        )


def _wrap_native_error(e: Exception) -> HebbsError:
    """Convert native extension errors to HEBBS exceptions."""
    msg = str(e)
    if "not found" in msg.lower():
        ulid = ""
        if ":" in msg:
            ulid = msg.split(":")[-1].strip()
        return MemoryNotFoundError(ulid)
    if "invalid input" in msg.lower() or "invalid" in msg.lower():
        return InvalidInputError("", "", msg)
    if "storage" in msg.lower():
        return StorageError(msg)
    if "embedding" in msg.lower():
        return EmbeddingError(msg)
    return HebbsError(msg)


class NativeBackend:
    """Embedded mode backend using the PyO3 native extension."""

    def __init__(
        self,
        data_dir: str,
        use_mock_embedder: bool = True,
        embedding_dimensions: int = 384,
    ) -> None:
        native = _import_native()
        try:
            self._engine = native.NativeEngine(
                data_dir=data_dir,
                use_mock_embedder=use_mock_embedder,
                embedding_dimensions=embedding_dimensions,
            )
        except Exception as e:
            raise _wrap_native_error(e) from e

    def remember(
        self,
        content: str,
        importance: float = 0.5,
        context: dict[str, Any] | None = None,
        entity_id: str | None = None,
    ) -> Memory:
        try:
            result = self._engine.remember(
                content=content,
                importance=importance,
                context=context,
                entity_id=entity_id,
            )
            return Memory._from_dict(result)
        except Exception as e:
            raise _wrap_native_error(e) from e

    def get(self, memory_id: str) -> Memory:
        try:
            result = self._engine.get(memory_id)
            return Memory._from_dict(result)
        except Exception as e:
            raise _wrap_native_error(e) from e

    def recall(
        self,
        cue: str,
        strategy: str = "similarity",
        top_k: int = 10,
        entity_id: str | None = None,
        max_depth: int | None = None,
        time_range: tuple[int, int] | None = None,
        strategies: list[str] | None = None,
    ) -> RecallOutput:
        try:
            result = self._engine.recall(
                cue=cue,
                strategy=strategy,
                top_k=top_k,
                entity_id=entity_id,
                max_depth=max_depth,
                time_range=time_range,
                strategies=strategies,
            )
            return RecallOutput._from_dict(result)
        except Exception as e:
            raise _wrap_native_error(e) from e

    def revise(
        self,
        memory_id: str,
        content: str | None = None,
        importance: float | None = None,
        context: dict[str, Any] | None = None,
        context_mode: str = "merge",
        entity_id: str | None = None,
    ) -> Memory:
        try:
            result = self._engine.revise(
                memory_id=memory_id,
                content=content,
                importance=importance,
                context=context,
                context_mode=context_mode,
                entity_id=entity_id,
            )
            return Memory._from_dict(result)
        except Exception as e:
            raise _wrap_native_error(e) from e

    def forget(
        self,
        memory_ids: list[str] | None = None,
        entity_id: str | None = None,
        staleness_threshold_us: int | None = None,
        access_count_floor: int | None = None,
        memory_kind: str | None = None,
        decay_score_floor: float | None = None,
    ) -> ForgetOutput:
        try:
            result = self._engine.forget(
                memory_ids=memory_ids,
                entity_id=entity_id,
                staleness_threshold_us=staleness_threshold_us,
                access_count_floor=access_count_floor,
                memory_kind=memory_kind,
                decay_score_floor=decay_score_floor,
            )
            return ForgetOutput._from_dict(result)
        except Exception as e:
            raise _wrap_native_error(e) from e

    def prime(
        self,
        entity_id: str,
        max_memories: int = 50,
        similarity_cue: str | None = None,
    ) -> PrimeOutput:
        try:
            result = self._engine.prime(
                entity_id=entity_id,
                max_memories=max_memories,
                similarity_cue=similarity_cue,
            )
            return PrimeOutput._from_dict(result)
        except Exception as e:
            raise _wrap_native_error(e) from e

    def subscribe(
        self,
        entity_id: str | None = None,
        confidence_threshold: float = 0.6,
    ) -> "NativeSubscribeStream":
        try:
            sub = self._engine.subscribe(
                entity_id=entity_id,
                confidence_threshold=confidence_threshold,
            )
            return NativeSubscribeStream(sub)
        except Exception as e:
            raise _wrap_native_error(e) from e

    def reflect(
        self,
        entity_id: str | None = None,
        since_us: int | None = None,
    ) -> ReflectOutput:
        try:
            result = self._engine.reflect(
                entity_id=entity_id,
                since_us=since_us,
            )
            return ReflectOutput._from_dict(result)
        except Exception as e:
            raise _wrap_native_error(e) from e

    def insights(
        self,
        entity_id: str | None = None,
        min_confidence: float | None = None,
        max_results: int | None = None,
    ) -> list[Memory]:
        try:
            result = self._engine.insights(
                entity_id=entity_id,
                min_confidence=min_confidence,
                max_results=max_results,
            )
            return [Memory._from_dict(m) for m in result]
        except Exception as e:
            raise _wrap_native_error(e) from e

    def count(self) -> int:
        try:
            return self._engine.count()
        except Exception as e:
            raise _wrap_native_error(e) from e

    def close(self) -> None:
        try:
            self._engine.close()
        except Exception:
            pass


class NativeSubscribeStream:
    """Iterator wrapper for embedded mode subscriptions."""

    def __init__(self, native_sub: Any) -> None:
        self._sub = native_sub
        self._closed = False

    def feed(self, text: str) -> None:
        if self._closed:
            raise HebbsError("subscription is closed")
        try:
            self._sub.feed(text)
        except Exception as e:
            raise _wrap_native_error(e) from e

    def poll(self, timeout_secs: float | None = None) -> SubscribePush | None:
        if self._closed:
            return None
        try:
            if timeout_secs is not None:
                result = self._sub.poll_timeout(timeout_secs)
            else:
                result = self._sub.poll()
            if result is None:
                return None
            return SubscribePush._from_dict(result)
        except Exception as e:
            raise _wrap_native_error(e) from e

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                self._sub.close()
            except Exception:
                pass

    def __enter__(self) -> "NativeSubscribeStream":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __iter__(self) -> Iterator[SubscribePush]:
        return self

    def __next__(self) -> SubscribePush:
        while not self._closed:
            push = self.poll(timeout_secs=1.0)
            if push is not None:
                return push
        raise StopIteration

    def __del__(self) -> None:
        self.close()
