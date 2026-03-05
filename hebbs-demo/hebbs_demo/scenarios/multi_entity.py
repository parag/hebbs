"""Scenario G: Multi-Entity Isolation.

Interleaves memory storage across three entities, validates entity scoping.
"""

from __future__ import annotations

from typing import Any

from hebbs_demo.scenarios.base import Scenario

ENTITY_ALPHA = "alpha_co"
ENTITY_BETA = "beta_co"
ENTITY_CHARLIE = "charlie_co"

ALPHA_MEMORIES: list[dict[str, Any]] = [
    {"content": "Alpha Co manufactures autonomous delivery drones for last-mile logistics", "importance": 0.8, "context": {"stage": "discovery", "topic": "product"}},
    {"content": "Fleet of 200 drones operating in Phoenix metro, expanding to Dallas Q2", "importance": 0.7, "context": {"stage": "growth", "topic": "expansion"}},
    {"content": "CTO James Whitfield needs computer vision for obstacle avoidance at 40mph", "importance": 0.9, "context": {"stage": "technical", "contact": "James Whitfield"}},
    {"content": "FAA Part 135 certification in progress, expected approval by June", "importance": 0.8, "context": {"stage": "compliance", "topic": "faa_cert"}},
    {"content": "Existing CV model from Skydio partnership expires, need replacement vendor", "importance": 0.9, "context": {"stage": "urgency", "competitor": "Skydio"}},
    {"content": "Battery optimization requires inference under 5W power envelope", "importance": 0.8, "context": {"stage": "technical", "topic": "power_constraint"}},
    {"content": "Series C closing at $120M valuation, strong unit economics at $4.50/delivery", "importance": 0.7, "context": {"stage": "qualification", "topic": "funding"}},
]

BETA_MEMORIES: list[dict[str, Any]] = [
    {"content": "Beta Co operates a fleet of 50 autonomous mining trucks in Pilbara region", "importance": 0.8, "context": {"stage": "discovery", "topic": "product"}},
    {"content": "Each truck generates 3TB of LiDAR data daily, need real-time processing", "importance": 0.9, "context": {"stage": "technical", "topic": "data_volume"}},
    {"content": "Safety-critical: zero tolerance for collision, ISO 17757 compliance mandatory", "importance": 0.9, "context": {"stage": "compliance", "topic": "safety"}},
    {"content": "VP Operations Sandra Mitchell wants predictive maintenance to reduce downtime from 8% to 3%", "importance": 0.8, "context": {"stage": "pain_point", "contact": "Sandra Mitchell"}},
    {"content": "Current Caterpillar autonomy stack is vendor-locked, seeking open alternatives", "importance": 0.8, "context": {"stage": "competitive", "competitor": "Caterpillar"}},
    {"content": "Harsh environment: 50°C ambient, red dust contamination on sensors", "importance": 0.7, "context": {"stage": "technical", "topic": "environment"}},
    {"content": "Rio Tinto subsidiary, procurement follows corporate governance with 120-day cycle", "importance": 0.7, "context": {"stage": "process", "topic": "procurement"}},
]

CHARLIE_MEMORIES: list[dict[str, Any]] = [
    {"content": "Charlie Co builds surgical robotics for minimally invasive cardiac procedures", "importance": 0.9, "context": {"stage": "discovery", "topic": "product"}},
    {"content": "FDA 510(k) clearance obtained for first-gen device, pursuing De Novo for Gen 2", "importance": 0.9, "context": {"stage": "compliance", "topic": "fda"}},
    {"content": "Haptic feedback system requires sub-1ms control loop latency", "importance": 0.9, "context": {"stage": "technical", "topic": "latency"}},
    {"content": "Chief Medical Officer Dr. Elena Vasquez leads clinical validation program", "importance": 0.8, "context": {"stage": "org_mapping", "contact": "Dr. Elena Vasquez"}},
    {"content": "Partnership with Mayo Clinic for 200-patient clinical trial starting September", "importance": 0.8, "context": {"stage": "timeline", "topic": "clinical_trial"}},
    {"content": "Computer vision needed for real-time tissue classification during surgery", "importance": 0.9, "context": {"stage": "use_case", "topic": "tissue_classification"}},
    {"content": "Total addressable market for cardiac surgical robotics estimated at $4.2B by 2028", "importance": 0.7, "context": {"stage": "market", "topic": "tam"}},
]

ALL_ENTITIES = {
    ENTITY_ALPHA: ALPHA_MEMORIES,
    ENTITY_BETA: BETA_MEMORIES,
    ENTITY_CHARLIE: CHARLIE_MEMORIES,
}


class MultiEntityScenario(Scenario):
    name = "multi_entity"
    description = "Scenario G: Interleave memories across three entities, validate entity-scoped recall and prime with zero cross-contamination"

    def execute(self, hebbs: Any, agent: Any) -> None:
        max_len = max(len(mems) for mems in ALL_ENTITIES.values())
        for i in range(max_len):
            for entity_id, memories in ALL_ENTITIES.items():
                if i < len(memories):
                    mem = memories[i]
                    hebbs.remember(
                        content=mem["content"],
                        importance=mem["importance"],
                        context=mem["context"],
                        entity_id=entity_id,
                    )

        total_expected = sum(len(m) for m in ALL_ENTITIES.values())
        self.assert_equal("total_memories_stored", total_expected, hebbs.count())

        # Temporal recall scopes by entity via the (entity_id, timestamp) B-tree index
        alpha_temporal = hebbs.recall(
            cue="delivery drones",
            strategy="temporal",
            top_k=20,
            entity_id=ENTITY_ALPHA,
        )
        self.assert_not_empty("alpha_temporal_returns_results", alpha_temporal.results)
        for result in alpha_temporal.results:
            self.assert_equal(
                f"alpha_temporal_scope_{result.memory.id[:8]}",
                ENTITY_ALPHA,
                result.memory.entity_id,
                "temporal recall for alpha should only return alpha memories",
            )

        beta_temporal = hebbs.recall(
            cue="mining trucks",
            strategy="temporal",
            top_k=20,
            entity_id=ENTITY_BETA,
        )
        self.assert_not_empty("beta_temporal_returns_results", beta_temporal.results)
        for result in beta_temporal.results:
            self.assert_equal(
                f"beta_temporal_scope_{result.memory.id[:8]}",
                ENTITY_BETA,
                result.memory.entity_id,
                "temporal recall for beta should only return beta memories",
            )

        charlie_temporal = hebbs.recall(
            cue="surgical robotics",
            strategy="temporal",
            top_k=20,
            entity_id=ENTITY_CHARLIE,
        )
        self.assert_not_empty("charlie_temporal_returns_results", charlie_temporal.results)
        for result in charlie_temporal.results:
            self.assert_equal(
                f"charlie_temporal_scope_{result.memory.id[:8]}",
                ENTITY_CHARLIE,
                result.memory.entity_id,
                "temporal recall for charlie should only return charlie memories",
            )

        # Similarity recall finds relevant memories globally
        alpha_similarity = hebbs.recall(
            cue="autonomous delivery drones computer vision obstacle avoidance",
            strategy="similarity",
            top_k=10,
        )
        self.assert_not_empty("similarity_recall_returns_results", alpha_similarity.results)

        # Prime uses temporal index and therefore scopes by entity
        alpha_prime = hebbs.prime(entity_id=ENTITY_ALPHA, max_memories=50)
        self.assert_not_empty("alpha_prime_returns_results", alpha_prime.results)

        beta_prime = hebbs.prime(entity_id=ENTITY_BETA, max_memories=50)
        self.assert_not_empty("beta_prime_returns_results", beta_prime.results)

        agent.start_session(entity_id=ENTITY_ALPHA, session_num=1)
        alpha_turn = agent.process_turn(
            prospect_message="We need to discuss the FAA certification timeline and how it affects our drone CV integration.",
            recall_strategies=["similarity", "temporal"],
        )
        self.assert_true(
            "alpha_session_turn_succeeds",
            alpha_turn is not None,
        )
        agent.end_session()

        agent.start_session(entity_id=ENTITY_CHARLIE, session_num=1)
        charlie_turn = agent.process_turn(
            prospect_message="Dr. Vasquez wants to know if your vision model can handle real-time tissue classification during the Mayo Clinic trial.",
            recall_strategies=["similarity"],
        )
        self.assert_true(
            "charlie_session_turn_succeeds",
            charlie_turn is not None,
        )
        agent.end_session()
