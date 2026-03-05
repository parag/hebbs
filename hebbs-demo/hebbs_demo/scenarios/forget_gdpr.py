"""Scenario F: Forget & GDPR Right-to-Erasure.

Stores 20 memories for a single prospect entity, verifies they are
retrievable, issues a full entity forget, and confirms complete
erasure — validating GDPR Article 17 compliance flow.
"""

from __future__ import annotations

from typing import Any

from hebbs_demo.scenarios.base import Scenario

ENTITY_ID = "prospect_x"

PROSPECT_MEMORIES: list[dict[str, Any]] = [
    {"content": "Prospect X is a mid-size insurance carrier writing $400M in annual premiums", "importance": 0.7, "context": {"stage": "discovery", "topic": "company_profile"}},
    {"content": "CEO David Park wants to modernize claims processing from 14-day to same-day", "importance": 0.9, "context": {"stage": "discovery", "contact": "David Park", "role": "CEO"}},
    {"content": "Current claims system built on AS/400, migrating to cloud-native stack", "importance": 0.8, "context": {"stage": "technical", "topic": "legacy_migration"}},
    {"content": "Fraud detection gap: estimated $12M annual leakage from staged auto claims", "importance": 0.9, "context": {"stage": "pain_point", "topic": "fraud", "amount": "$12M"}},
    {"content": "Head of Claims Lisa Wong champions AI adoption, reports directly to CEO", "importance": 0.8, "context": {"stage": "org_mapping", "contact": "Lisa Wong", "role": "Head of Claims"}},
    {"content": "IT team of 45, DevOps maturity level 3, using Terraform and ArgoCD", "importance": 0.6, "context": {"stage": "technical", "topic": "team_capabilities"}},
    {"content": "State regulatory filings require model explainability for claim denials", "importance": 0.9, "context": {"stage": "compliance", "topic": "explainability"}},
    {"content": "Competitor Verisk quoted $2.1M annually, Prospect X found it expensive", "importance": 0.8, "context": {"stage": "competitive", "competitor": "Verisk", "price": "$2.1M"}},
    {"content": "POC success criteria: process 1000 sample claims with 95% accuracy", "importance": 0.8, "context": {"stage": "evaluation", "topic": "poc_criteria"}},
    {"content": "Data warehouse holds 8 years of claims history, 4.2M records total", "importance": 0.7, "context": {"stage": "technical", "topic": "data_volume"}},
    {"content": "Board meeting in March to decide on vendor selection for claims AI", "importance": 0.9, "context": {"stage": "timeline", "topic": "decision_date"}},
    {"content": "HIPAA compliance required for health insurance line of business", "importance": 0.9, "context": {"stage": "compliance", "topic": "hipaa"}},
    {"content": "Prospect X processes 85K claims per month across auto, home, and health", "importance": 0.8, "context": {"stage": "qualification", "topic": "volume"}},
    {"content": "Integration requirement: must connect to Guidewire ClaimCenter via REST API", "importance": 0.7, "context": {"stage": "technical", "topic": "integration"}},
    {"content": "CFO wants payback period under 18 months, total budget capped at $1.5M", "importance": 0.8, "context": {"stage": "qualification", "topic": "budget", "cap": "$1.5M"}},
    {"content": "Subrogation recovery team manually reviews 2K cases monthly, wants automation", "importance": 0.7, "context": {"stage": "use_case", "topic": "subrogation"}},
    {"content": "Natural disaster surge handling: 10x volume spikes during hurricane season", "importance": 0.8, "context": {"stage": "requirement", "topic": "scalability"}},
    {"content": "Data residency: all PII must remain in US-East region per corporate policy", "importance": 0.8, "context": {"stage": "compliance", "topic": "data_residency"}},
    {"content": "Agent desktop integration needed — adjusters use Salesforce Service Cloud", "importance": 0.6, "context": {"stage": "technical", "topic": "ux_integration"}},
    {"content": "Prospect X's parent company Meridian Holdings may expand deal to 3 subsidiaries", "importance": 0.9, "context": {"stage": "expansion", "topic": "upsell_potential"}},
]


class ForgetGdprScenario(Scenario):
    name = "forget_gdpr"
    description = "Scenario F: Store 20 prospect memories, verify retrieval, execute full entity forget, confirm complete GDPR-compliant erasure"

    def execute(self, hebbs: Any, agent: Any) -> None:
        for mem in PROSPECT_MEMORIES:
            hebbs.remember(
                content=mem["content"],
                importance=mem["importance"],
                context=mem["context"],
                entity_id=ENTITY_ID,
            )

        count_after_store = hebbs.count()
        self.assert_equal("memories_stored", len(PROSPECT_MEMORIES), count_after_store)

        pre_recall = hebbs.recall(
            cue="insurance claims processing fraud detection",
            strategy="similarity",
            top_k=20,
            entity_id=ENTITY_ID,
        )
        self.assert_not_empty(
            "pre_forget_recall_returns_results",
            pre_recall.results,
            "recall should find memories before forget",
        )

        recalled_count = len(pre_recall.results)
        self.assert_gte(
            "pre_forget_recall_count",
            recalled_count,
            1,
            "at least one memory should be recalled",
        )

        forget_result = hebbs.forget(entity_id=ENTITY_ID)
        self.assert_true(
            "forget_returns_result",
            forget_result is not None,
            "forget() should return a ForgetOutput",
        )
        self.assert_gte(
            "forget_count_matches",
            forget_result.forgotten_count,
            len(PROSPECT_MEMORIES),
            "forgotten_count should be at least the number of memories stored",
        )

        post_recall = hebbs.recall(
            cue="insurance claims processing fraud detection",
            strategy="similarity",
            top_k=20,
            entity_id=ENTITY_ID,
        )
        self.assert_empty(
            "post_forget_recall_empty",
            post_recall.results,
            "recall should return no results after entity forget",
        )

        prime_result = hebbs.prime(entity_id=ENTITY_ID, max_memories=50)
        self.assert_empty(
            "post_forget_prime_empty",
            prime_result.results,
            "prime should return no results after entity forget",
        )
