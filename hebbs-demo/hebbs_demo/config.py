"""Configuration loader: TOML files, env-var resolution, validated defaults."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


@dataclass
class LlmProviderConfig:
    api_key_env: str = ""
    model: str = ""
    base_url: str = ""

    @property
    def api_key(self) -> str | None:
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        return None


@dataclass
class EmbeddingProviderConfig:
    api_key_env: str = ""
    model: str = ""
    base_url: str = ""

    @property
    def api_key(self) -> str | None:
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        return None


@dataclass
class LlmConfig:
    conversation_provider: str = "openai"
    conversation_model: str = "gpt-4o"
    extraction_provider: str = "openai"
    extraction_model: str = "gpt-4o-mini"
    openai: LlmProviderConfig = field(default_factory=lambda: LlmProviderConfig(
        api_key_env="OPENAI_API_KEY", model="gpt-4o",
    ))
    anthropic: LlmProviderConfig = field(default_factory=lambda: LlmProviderConfig(
        api_key_env="ANTHROPIC_API_KEY", model="claude-sonnet-4-20250514",
    ))
    ollama: LlmProviderConfig = field(default_factory=lambda: LlmProviderConfig(
        base_url="http://localhost:11434", model="llama3.2",
    ))


@dataclass
class EmbeddingConfig:
    provider: str = "onnx"
    onnx: EmbeddingProviderConfig = field(default_factory=lambda: EmbeddingProviderConfig(
        model="bge-small-en-v1.5",
    ))
    openai: EmbeddingProviderConfig = field(default_factory=lambda: EmbeddingProviderConfig(
        api_key_env="OPENAI_API_KEY", model="text-embedding-3-small",
    ))
    ollama: EmbeddingProviderConfig = field(default_factory=lambda: EmbeddingProviderConfig(
        base_url="http://localhost:11434", model="nomic-embed-text",
    ))


@dataclass
class ReflectConfig:
    provider: str = "openai"
    model: str = "gpt-4o"
    threshold: int = 20


@dataclass
class HebbsConfig:
    data_dir: str = "./hebbs-data"
    reflect: ReflectConfig = field(default_factory=ReflectConfig)


@dataclass
class DemoConfig:
    llm: LlmConfig = field(default_factory=LlmConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    hebbs: HebbsConfig = field(default_factory=HebbsConfig)

    @classmethod
    def from_toml(cls, path: str | Path) -> DemoConfig:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        return cls._from_dict(raw)

    @classmethod
    def default(cls) -> DemoConfig:
        return cls()

    @classmethod
    def _from_dict(cls, d: dict[str, Any]) -> DemoConfig:
        cfg = cls()

        llm_raw = d.get("llm", {})
        cfg.llm.conversation_provider = llm_raw.get("conversation_provider", cfg.llm.conversation_provider)
        cfg.llm.conversation_model = llm_raw.get("conversation_model", cfg.llm.conversation_model)
        cfg.llm.extraction_provider = llm_raw.get("extraction_provider", cfg.llm.extraction_provider)
        cfg.llm.extraction_model = llm_raw.get("extraction_model", cfg.llm.extraction_model)

        for provider_name in ("openai", "anthropic", "ollama"):
            prov_raw = llm_raw.get(provider_name, {})
            prov_cfg = getattr(cfg.llm, provider_name)
            if "api_key_env" in prov_raw:
                prov_cfg.api_key_env = prov_raw["api_key_env"]
            if "model" in prov_raw:
                prov_cfg.model = prov_raw["model"]
            if "base_url" in prov_raw:
                prov_cfg.base_url = prov_raw["base_url"]

        embed_raw = d.get("embedding", {})
        cfg.embedding.provider = embed_raw.get("provider", cfg.embedding.provider)
        for provider_name in ("onnx", "openai", "ollama"):
            prov_raw = embed_raw.get(provider_name, {})
            prov_cfg = getattr(cfg.embedding, provider_name)
            if "api_key_env" in prov_raw:
                prov_cfg.api_key_env = prov_raw["api_key_env"]
            if "model" in prov_raw:
                prov_cfg.model = prov_raw["model"]
            if "base_url" in prov_raw:
                prov_cfg.base_url = prov_raw["base_url"]

        hebbs_raw = d.get("hebbs", {})
        cfg.hebbs.data_dir = hebbs_raw.get("data_dir", cfg.hebbs.data_dir)
        reflect_raw = hebbs_raw.get("reflect", {})
        cfg.hebbs.reflect.provider = reflect_raw.get("provider", cfg.hebbs.reflect.provider)
        cfg.hebbs.reflect.model = reflect_raw.get("model", cfg.hebbs.reflect.model)
        cfg.hebbs.reflect.threshold = reflect_raw.get("threshold", cfg.hebbs.reflect.threshold)

        return cfg

    def get_llm_provider_config(self, provider_name: str) -> LlmProviderConfig:
        return getattr(self.llm, provider_name, LlmProviderConfig())

    def validate(self) -> list[str]:
        """Return a list of validation warnings (empty = all good)."""
        warnings: list[str] = []
        conv_prov = self.llm.conversation_provider
        if conv_prov in ("openai", "anthropic"):
            prov_cfg = self.get_llm_provider_config(conv_prov)
            if not prov_cfg.api_key:
                warnings.append(
                    f"${prov_cfg.api_key_env} not set — "
                    f"conversation LLM ({conv_prov}) will fail"
                )
        ext_prov = self.llm.extraction_provider
        if ext_prov in ("openai", "anthropic"):
            prov_cfg = self.get_llm_provider_config(ext_prov)
            if not prov_cfg.api_key:
                env_name = prov_cfg.api_key_env
                if env_name not in [w.split()[0].strip("$") for w in warnings]:
                    warnings.append(
                        f"${env_name} not set — "
                        f"extraction LLM ({ext_prov}) will fail"
                    )
        return warnings
