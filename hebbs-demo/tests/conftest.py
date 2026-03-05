"""Shared pytest fixtures for hebbs-demo tests."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, Generator

import pytest

from hebbs_demo.config import DemoConfig
from hebbs_demo.display import DisplayManager, Verbosity
from hebbs_demo.llm_client import MockLlmClient


@pytest.fixture
def config() -> DemoConfig:
    return DemoConfig.default()


@pytest.fixture
def mock_llm(config: DemoConfig) -> MockLlmClient:
    return MockLlmClient(config)


@pytest.fixture
def display() -> DisplayManager:
    from rich.console import Console
    return DisplayManager(Verbosity.QUIET, Console(quiet=True))


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    d = tempfile.mkdtemp(prefix="hebbs_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def hebbs_engine(temp_dir: str) -> Generator[Any, None, None]:
    """Create a fresh HEBBS engine with mock embedder for testing."""
    try:
        from hebbs import HEBBS
        engine = HEBBS.open(
            data_dir=temp_dir,
            use_mock_embedder=True,
            embedding_dimensions=384,
        )
        yield engine
        engine.close()
    except ImportError:
        pytest.skip("HEBBS native extension not available")
