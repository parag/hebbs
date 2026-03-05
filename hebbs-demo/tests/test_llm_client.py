"""Unit tests for the LLM client abstraction."""

from __future__ import annotations

import json

import pytest

from hebbs_demo.config import DemoConfig
from hebbs_demo.llm_client import LlmClient, LlmResponse, LlmStats, MockLlmClient


class TestLlmResponse:
    def test_fields(self):
        r = LlmResponse(
            content="hello", input_tokens=10, output_tokens=20,
            latency_ms=5.0, model="test", provider="test",
        )
        assert r.content == "hello"
        assert r.input_tokens == 10
        assert r.output_tokens == 20
        assert r.latency_ms == 5.0


class TestLlmStats:
    def test_record(self):
        stats = LlmStats()
        r = LlmResponse(content="x", input_tokens=100, output_tokens=50, latency_ms=10)
        stats.record(r, "conversation")
        assert stats.total_calls == 1
        assert stats.total_input_tokens == 100
        assert stats.total_output_tokens == 50
        assert stats.calls_by_role["conversation"] == 1

    def test_estimated_cost(self):
        stats = LlmStats()
        r = LlmResponse(content="x", input_tokens=1_000_000, output_tokens=1_000_000, latency_ms=10)
        stats.record(r, "test")
        assert stats.estimated_cost_usd == pytest.approx(12.5, rel=0.01)

    def test_multiple_records(self):
        stats = LlmStats()
        for i in range(5):
            r = LlmResponse(content="x", input_tokens=10, output_tokens=5, latency_ms=i)
            stats.record(r, "conversation")
        assert stats.total_calls == 5
        assert stats.total_input_tokens == 50
        assert stats.total_output_tokens == 25
        assert stats.total_latency_ms == pytest.approx(10.0)


class TestMockLlmClient:
    def test_conversation(self):
        client = MockLlmClient()
        messages = [{"role": "user", "content": "Hello, how are you?"}]
        resp = client.conversation(messages)
        assert resp.content
        assert resp.provider == "mock"
        assert resp.latency_ms > 0

    def test_extraction(self):
        client = MockLlmClient()
        messages = [
            {"role": "system", "content": "You are a memory extraction system."},
            {"role": "user", "content": "Prospect said they need SOC 2."},
        ]
        resp = client.extract_memories(messages)
        data = json.loads(resp.content)
        assert "memories" in data
        assert len(data["memories"]) > 0

    def test_stats_tracking(self):
        client = MockLlmClient()
        messages = [{"role": "user", "content": "Hello"}]
        client.conversation(messages)
        client.conversation(messages)
        client.extract_memories(messages)
        assert client.stats.total_calls == 3
        assert client.stats.calls_by_role.get("conversation", 0) == 2
        assert client.stats.calls_by_role.get("extraction", 0) == 1

    def test_simulate_prospect(self):
        client = MockLlmClient()
        messages = [{"role": "user", "content": "Tell me about your pricing."}]
        resp = client.simulate_prospect(messages)
        assert resp.content
        assert resp.provider == "mock"

    def test_dispatch_unknown_provider(self):
        client = LlmClient(DemoConfig())
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            client._dispatch("nonexistent", "model", [{"role": "user", "content": "hi"}])
