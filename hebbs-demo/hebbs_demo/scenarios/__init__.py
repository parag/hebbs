"""Scripted scenario suite for validating HEBBS capabilities."""

from hebbs_demo.scenarios.base import Scenario, ScenarioResult, Assertion
from hebbs_demo.scenarios.discovery_call import DiscoveryCallScenario
from hebbs_demo.scenarios.objection_handling import ObjectionHandlingScenario
from hebbs_demo.scenarios.multi_session import MultiSessionScenario
from hebbs_demo.scenarios.reflect_learning import ReflectLearningScenario
from hebbs_demo.scenarios.subscribe_realtime import SubscribeRealtimeScenario
from hebbs_demo.scenarios.forget_gdpr import ForgetGdprScenario
from hebbs_demo.scenarios.multi_entity import MultiEntityScenario

ALL_SCENARIOS: dict[str, type[Scenario]] = {
    "discovery_call": DiscoveryCallScenario,
    "objection_handling": ObjectionHandlingScenario,
    "multi_session": MultiSessionScenario,
    "reflect_learning": ReflectLearningScenario,
    "subscribe_realtime": SubscribeRealtimeScenario,
    "forget_gdpr": ForgetGdprScenario,
    "multi_entity": MultiEntityScenario,
}

__all__ = [
    "Scenario",
    "ScenarioResult",
    "Assertion",
    "ALL_SCENARIOS",
    "DiscoveryCallScenario",
    "ObjectionHandlingScenario",
    "MultiSessionScenario",
    "ReflectLearningScenario",
    "SubscribeRealtimeScenario",
    "ForgetGdprScenario",
    "MultiEntityScenario",
]
