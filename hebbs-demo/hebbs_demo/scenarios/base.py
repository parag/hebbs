"""Base class for scripted scenarios with assertion framework.

Every scenario:
  1. Gets a fresh HEBBS engine instance (temp directory)
  2. Runs a predefined sequence of operations
  3. Validates assertions about HEBBS behavior
  4. Reports pass/fail with details
"""

from __future__ import annotations

import shutil
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console

from hebbs_demo.agent import SalesAgent
from hebbs_demo.config import DemoConfig
from hebbs_demo.display import DisplayManager, Verbosity


@dataclass
class Assertion:
    name: str
    passed: bool
    message: str = ""
    expected: Any = None
    actual: Any = None


@dataclass
class ScenarioResult:
    name: str
    passed: bool
    assertions: list[Assertion] = field(default_factory=list)
    elapsed_ms: float = 0.0
    error: str | None = None

    @property
    def failed_assertions(self) -> list[Assertion]:
        return [a for a in self.assertions if not a.passed]


def _open_hebbs(data_dir: str, use_mock: bool = True) -> Any:
    """Open a HEBBS engine in embedded mode."""
    from hebbs import HEBBS
    return HEBBS.open(
        data_dir=data_dir,
        use_mock_embedder=use_mock,
        embedding_dimensions=384,
    )


class Scenario(ABC):
    """Base class for scripted scenarios."""

    name: str = "unnamed"
    description: str = ""

    def __init__(
        self,
        config: DemoConfig | None = None,
        verbosity: Verbosity = Verbosity.NORMAL,
        use_mock_llm: bool = True,
        use_mock_embedder: bool = True,
        console: Console | None = None,
    ) -> None:
        self._config = config or DemoConfig()
        self._verbosity = verbosity
        self._use_mock_llm = use_mock_llm
        self._use_mock_embedder = use_mock_embedder
        self._console = console or Console()
        self._assertions: list[Assertion] = []
        self._temp_dir: str | None = None

    def assert_true(self, name: str, condition: bool, message: str = "") -> None:
        self._assertions.append(Assertion(
            name=name, passed=condition, message=message,
        ))

    def assert_equal(self, name: str, expected: Any, actual: Any, message: str = "") -> None:
        self._assertions.append(Assertion(
            name=name, passed=(expected == actual),
            message=message or f"expected {expected}, got {actual}",
            expected=expected, actual=actual,
        ))

    def assert_gte(self, name: str, actual: int | float, minimum: int | float, message: str = "") -> None:
        self._assertions.append(Assertion(
            name=name, passed=(actual >= minimum),
            message=message or f"expected >= {minimum}, got {actual}",
            expected=f">= {minimum}", actual=actual,
        ))

    def assert_empty(self, name: str, collection: Any, message: str = "") -> None:
        is_empty = len(collection) == 0 if collection is not None else True
        self._assertions.append(Assertion(
            name=name, passed=is_empty,
            message=message or f"expected empty, got {len(collection) if collection else 0} items",
        ))

    def assert_not_empty(self, name: str, collection: Any, message: str = "") -> None:
        is_not_empty = len(collection) > 0 if collection is not None else False
        self._assertions.append(Assertion(
            name=name, passed=is_not_empty,
            message=message or "expected non-empty collection",
        ))

    def _setup(self) -> tuple[Any, SalesAgent]:
        """Create a fresh HEBBS engine and SalesAgent in a temp directory."""
        self._temp_dir = tempfile.mkdtemp(prefix="hebbs_scenario_")
        hebbs = _open_hebbs(self._temp_dir, use_mock=self._use_mock_embedder)
        display = DisplayManager(self._verbosity, self._console)
        agent = SalesAgent(
            config=self._config,
            hebbs=hebbs,
            display=display,
            use_mock_llm=self._use_mock_llm,
        )
        return hebbs, agent

    def _teardown(self, hebbs: Any) -> None:
        """Clean up the HEBBS engine and temp directory."""
        try:
            hebbs.close()
        except Exception:
            pass
        if self._temp_dir:
            try:
                shutil.rmtree(self._temp_dir, ignore_errors=True)
            except Exception:
                pass

    def run(self) -> ScenarioResult:
        """Execute the scenario and return results."""
        self._assertions = []
        t0 = time.perf_counter()

        try:
            hebbs, agent = self._setup()
        except Exception as e:
            return ScenarioResult(
                name=self.name, passed=False, elapsed_ms=0,
                error=f"Setup failed: {e}",
            )

        try:
            self.execute(hebbs, agent)
        except Exception as e:
            self._assertions.append(Assertion(
                name="scenario_execution", passed=False,
                message=f"Scenario raised exception: {e}",
            ))
        finally:
            self._teardown(hebbs)

        elapsed = (time.perf_counter() - t0) * 1000
        all_passed = all(a.passed for a in self._assertions)

        return ScenarioResult(
            name=self.name,
            passed=all_passed,
            assertions=list(self._assertions),
            elapsed_ms=elapsed,
        )

    @abstractmethod
    def execute(self, hebbs: Any, agent: SalesAgent) -> None:
        """Run the scenario's conversation and assertions.

        Subclasses implement this method. Use self.assert_* methods to record
        assertions that will be checked after execution.
        """
        ...
