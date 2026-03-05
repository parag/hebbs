"""Pytest suite running all 7 scenarios.

All tests use mock LLM and mock embedder by default (no API keys needed).
Tests marked with @pytest.mark.requires_openai need OPENAI_API_KEY.
"""

from __future__ import annotations

import os

import pytest
from rich.console import Console

from hebbs_demo.config import DemoConfig
from hebbs_demo.display import Verbosity
from hebbs_demo.scenarios import ALL_SCENARIOS


def _skip_if_no_hebbs():
    """Skip if HEBBS native extension is not available."""
    try:
        from hebbs import HEBBS
    except ImportError:
        pytest.skip("HEBBS native extension not available")


class TestDiscoveryCall:
    def test_passes_with_mock(self):
        _skip_if_no_hebbs()
        from hebbs_demo.scenarios.discovery_call import DiscoveryCallScenario
        scenario = DiscoveryCallScenario(
            verbosity=Verbosity.QUIET,
            use_mock_llm=True,
            console=Console(quiet=True),
        )
        result = scenario.run()
        assert result.passed, f"Failed assertions: {[a.name + ': ' + a.message for a in result.failed_assertions]}"
        assert len(result.assertions) >= 3


class TestObjectionHandling:
    def test_passes_with_mock(self):
        _skip_if_no_hebbs()
        from hebbs_demo.scenarios.objection_handling import ObjectionHandlingScenario
        scenario = ObjectionHandlingScenario(
            verbosity=Verbosity.QUIET,
            use_mock_llm=True,
            console=Console(quiet=True),
        )
        result = scenario.run()
        assert result.passed, f"Failed assertions: {[a.name + ': ' + a.message for a in result.failed_assertions]}"
        assert len(result.assertions) >= 3


class TestMultiSession:
    def test_passes_with_mock(self):
        _skip_if_no_hebbs()
        from hebbs_demo.scenarios.multi_session import MultiSessionScenario
        scenario = MultiSessionScenario(
            verbosity=Verbosity.QUIET,
            use_mock_llm=True,
            console=Console(quiet=True),
        )
        result = scenario.run()
        assert result.passed, f"Failed assertions: {[a.name + ': ' + a.message for a in result.failed_assertions]}"
        assert len(result.assertions) >= 3


class TestReflectLearning:
    def test_passes_with_mock(self):
        _skip_if_no_hebbs()
        from hebbs_demo.scenarios.reflect_learning import ReflectLearningScenario
        scenario = ReflectLearningScenario(
            verbosity=Verbosity.QUIET,
            use_mock_llm=True,
            console=Console(quiet=True),
        )
        result = scenario.run()
        assert result.passed, f"Failed assertions: {[a.name + ': ' + a.message for a in result.failed_assertions]}"
        assert len(result.assertions) >= 3


class TestSubscribeRealtime:
    def test_passes_with_mock(self):
        _skip_if_no_hebbs()
        from hebbs_demo.scenarios.subscribe_realtime import SubscribeRealtimeScenario
        scenario = SubscribeRealtimeScenario(
            verbosity=Verbosity.QUIET,
            use_mock_llm=True,
            console=Console(quiet=True),
        )
        result = scenario.run()
        assert result.passed, f"Failed assertions: {[a.name + ': ' + a.message for a in result.failed_assertions]}"
        assert len(result.assertions) >= 3


class TestForgetGdpr:
    def test_passes_with_mock(self):
        _skip_if_no_hebbs()
        from hebbs_demo.scenarios.forget_gdpr import ForgetGdprScenario
        scenario = ForgetGdprScenario(
            verbosity=Verbosity.QUIET,
            use_mock_llm=True,
            console=Console(quiet=True),
        )
        result = scenario.run()
        assert result.passed, f"Failed assertions: {[a.name + ': ' + a.message for a in result.failed_assertions]}"
        assert len(result.assertions) >= 4


class TestMultiEntity:
    def test_passes_with_mock(self):
        _skip_if_no_hebbs()
        from hebbs_demo.scenarios.multi_entity import MultiEntityScenario
        scenario = MultiEntityScenario(
            verbosity=Verbosity.QUIET,
            use_mock_llm=True,
            console=Console(quiet=True),
        )
        result = scenario.run()
        assert result.passed, f"Failed assertions: {[a.name + ': ' + a.message for a in result.failed_assertions]}"
        assert len(result.assertions) >= 3


class TestAllScenariosRegistered:
    def test_seven_scenarios_available(self):
        assert len(ALL_SCENARIOS) == 7

    def test_expected_names(self):
        expected = {
            "discovery_call", "objection_handling", "multi_session",
            "reflect_learning", "subscribe_realtime", "forget_gdpr", "multi_entity",
        }
        assert set(ALL_SCENARIOS.keys()) == expected
