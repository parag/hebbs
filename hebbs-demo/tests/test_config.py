"""Unit tests for config loading."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from hebbs_demo.config import DemoConfig


class TestDemoConfig:
    def test_default_config(self):
        cfg = DemoConfig.default()
        assert cfg.llm.conversation_provider == "openai"
        assert cfg.llm.conversation_model == "gpt-4o"
        assert cfg.llm.extraction_model == "gpt-4o-mini"
        assert cfg.embedding.provider == "onnx"
        assert cfg.hebbs.data_dir == "./hebbs-data"

    def test_from_toml(self, tmp_path: Path):
        toml_content = """
[llm]
conversation_provider = "anthropic"
conversation_model = "claude-sonnet-4-20250514"
extraction_provider = "openai"
extraction_model = "gpt-4o-mini"

[llm.anthropic]
api_key_env = "MY_KEY"

[embedding]
provider = "openai"

[hebbs]
data_dir = "/tmp/test"
"""
        f = tmp_path / "test.toml"
        f.write_text(toml_content)
        cfg = DemoConfig.from_toml(f)
        assert cfg.llm.conversation_provider == "anthropic"
        assert cfg.llm.anthropic.api_key_env == "MY_KEY"
        assert cfg.embedding.provider == "openai"
        assert cfg.hebbs.data_dir == "/tmp/test"

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            DemoConfig.from_toml("/nonexistent/path.toml")

    def test_validate_missing_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg = DemoConfig.default()
        warnings = cfg.validate()
        assert len(warnings) >= 1
        assert "OPENAI_API_KEY" in warnings[0]

    def test_validate_key_present(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        cfg = DemoConfig.default()
        warnings = cfg.validate()
        assert len(warnings) == 0

    def test_get_llm_provider_config(self):
        cfg = DemoConfig.default()
        openai_cfg = cfg.get_llm_provider_config("openai")
        assert openai_cfg.api_key_env == "OPENAI_API_KEY"
        unknown = cfg.get_llm_provider_config("nonexistent")
        assert unknown.api_key_env == ""

    def test_reflect_config(self, tmp_path: Path):
        toml_content = """
[hebbs.reflect]
provider = "ollama"
model = "llama3.2"
threshold = 50
"""
        f = tmp_path / "test.toml"
        f.write_text(toml_content)
        cfg = DemoConfig.from_toml(f)
        assert cfg.hebbs.reflect.provider == "ollama"
        assert cfg.hebbs.reflect.model == "llama3.2"
        assert cfg.hebbs.reflect.threshold == 50
