"""Unit tests for memory extraction prompt/parse logic."""

from __future__ import annotations

import json

import pytest

from hebbs_demo.memory_manager import (
    ExtractedMemory,
    ExtractionResult,
    parse_extraction_response,
)
from hebbs_demo.prompts import extraction_prompt, conversation_prompt


class TestParseExtractionResponse:
    def test_valid_json(self):
        raw = json.dumps({
            "memories": [
                {
                    "content": "Acme Corp needs SOC 2 compliance",
                    "importance": 0.8,
                    "context": {"company": "Acme Corp", "topic": "compliance"},
                    "edge_to_previous": False,
                }
            ],
            "skip_reason": None,
        })
        result = parse_extraction_response(raw)
        assert result.parse_success
        assert len(result.memories) == 1
        assert result.memories[0].content == "Acme Corp needs SOC 2 compliance"
        assert result.memories[0].importance == 0.8
        assert result.memories[0].context["company"] == "Acme Corp"

    def test_multiple_memories(self):
        raw = json.dumps({
            "memories": [
                {"content": "Fact 1", "importance": 0.9, "context": {}},
                {"content": "Fact 2", "importance": 0.7, "context": {"topic": "pricing"}},
                {"content": "Fact 3", "importance": 0.5, "context": {}},
            ],
            "skip_reason": None,
        })
        result = parse_extraction_response(raw)
        assert result.parse_success
        assert len(result.memories) == 3

    def test_skip_reason(self):
        raw = json.dumps({
            "memories": [],
            "skip_reason": "Small talk, no actionable information",
        })
        result = parse_extraction_response(raw)
        assert result.parse_success
        assert len(result.memories) == 0
        assert result.skip_reason == "Small talk, no actionable information"

    def test_empty_content_filtered(self):
        raw = json.dumps({
            "memories": [
                {"content": "", "importance": 0.5, "context": {}},
                {"content": "Valid fact", "importance": 0.7, "context": {}},
            ],
            "skip_reason": None,
        })
        result = parse_extraction_response(raw)
        assert len(result.memories) == 1
        assert result.memories[0].content == "Valid fact"

    def test_importance_clamped(self):
        raw = json.dumps({
            "memories": [
                {"content": "Over-important", "importance": 1.5, "context": {}},
                {"content": "Under-important", "importance": -0.3, "context": {}},
            ],
            "skip_reason": None,
        })
        result = parse_extraction_response(raw)
        assert result.memories[0].importance == 1.0
        assert result.memories[1].importance == 0.0

    def test_markdown_code_fence_stripped(self):
        raw = "```json\n" + json.dumps({
            "memories": [{"content": "Test", "importance": 0.6, "context": {}}],
            "skip_reason": None,
        }) + "\n```"
        result = parse_extraction_response(raw)
        assert result.parse_success
        assert len(result.memories) == 1

    def test_invalid_json(self):
        result = parse_extraction_response("this is not json at all")
        assert not result.parse_success
        assert len(result.memories) == 0

    def test_json_embedded_in_text(self):
        raw = 'Here is the extraction:\n{"memories": [{"content": "Test", "importance": 0.7, "context": {}}], "skip_reason": null}\nDone.'
        result = parse_extraction_response(raw)
        assert result.parse_success
        assert len(result.memories) == 1

    def test_invalid_context_type(self):
        raw = json.dumps({
            "memories": [
                {"content": "Test", "importance": 0.5, "context": "not a dict"},
            ],
            "skip_reason": None,
        })
        result = parse_extraction_response(raw)
        assert result.parse_success
        assert result.memories[0].context == {}

    def test_edge_to_previous(self):
        raw = json.dumps({
            "memories": [
                {"content": "Follow-up fact", "importance": 0.7, "context": {}, "edge_to_previous": True},
            ],
            "skip_reason": None,
        })
        result = parse_extraction_response(raw)
        assert result.memories[0].edge_to_previous is True

    def test_missing_optional_fields(self):
        raw = json.dumps({
            "memories": [{"content": "Minimal"}],
            "skip_reason": None,
        })
        result = parse_extraction_response(raw)
        assert result.parse_success
        assert result.memories[0].importance == 0.5
        assert result.memories[0].context == {}
        assert result.memories[0].edge_to_previous is False


class TestExtractionPrompt:
    def test_basic_structure(self):
        messages = extraction_prompt("I need compliance help", "Sure, let me help.")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "compliance" in messages[1]["content"]

    def test_with_entity_id(self):
        messages = extraction_prompt("Hi", "Hello", entity_id="acme_corp")
        assert "acme_corp" in messages[1]["content"]


class TestConversationPrompt:
    def test_basic_structure(self):
        messages = conversation_prompt(
            prospect_message="Tell me about your product",
            recalled_context="",
            session_history=[],
        )
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"
        assert "Tell me about your product" in messages[-1]["content"]

    def test_with_context(self):
        messages = conversation_prompt(
            prospect_message="What about pricing?",
            recalled_context="- Previous discussion about enterprise plan",
            session_history=[
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ],
            entity_id="beta_inc",
            insights="- Enterprise deals close faster in Q4",
        )
        assert len(messages) >= 4
        found_context = any("RECALLED MEMORIES" in m["content"] for m in messages)
        assert found_context

    def test_without_context(self):
        messages = conversation_prompt(
            prospect_message="Hello",
            recalled_context="",
            session_history=[],
        )
        assert len(messages) == 2
