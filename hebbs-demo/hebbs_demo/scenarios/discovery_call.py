"""Scenario A: Discovery Call — single-session memory formation and retrieval.

Simulates a 10-turn discovery call with a prospect (acme_corp).
Validates that HEBBS stores memories with context metadata and that
similarity recall surfaces relevant context mid-conversation.
"""

from __future__ import annotations

from typing import Any

from hebbs_demo.scenarios.base import Scenario

ENTITY = "acme_corp"

TURNS = [
    "Hi, I'm Sarah Chen, VP of Engineering at Acme Corp. We have about 200 developers across three offices.",
    "Our biggest pain point is knowledge silos. Teams in Berlin, Austin, and Singapore keep re-solving the same problems.",
    "We tried Confluence and Notion but adoption dropped off after a few months. People just stop updating the docs.",
    "What we really need is something that captures knowledge passively — without forcing engineers to write things down.",
    "We're also dealing with high attrition. When senior engineers leave, all their context walks out the door with them.",
    "Budget-wise, we have about $150K allocated for developer tooling this quarter. Is that in the right ballpark?",
    "Integration is critical. We're all-in on GitHub, Slack, and Linear. Anything that doesn't plug into those is a non-starter.",
    "Our CTO, Marcus, is the final decision-maker. He's very data-driven — he'll want to see measurable impact on onboarding time.",
    "Timeline is aggressive. We're hoping to pilot with one team in Q2 and roll out company-wide by end of Q3.",
    "Can you send over a technical architecture doc? Marcus will want to review the data residency and security model.",
]


class DiscoveryCallScenario(Scenario):
    name = "discovery_call"
    description = "Single-session discovery call validating memory formation and similarity recall"

    def execute(self, hebbs: Any, agent: Any) -> None:
        agent.start_session(entity_id=ENTITY, session_num=1)

        turn_results = []
        for i, message in enumerate(TURNS):
            strategies = ["similarity"] if i >= 3 else None
            result = agent.process_turn(message, recall_strategies=strategies)
            turn_results.append(result)

        agent.end_session()

        total_created = sum(t.memories_created for t in turn_results)
        self.assert_gte(
            "minimum_memories_stored", total_created, 5,
            f"Expected at least 5 memories from 10-turn call, got {total_created}",
        )

        late_recalls = sum(t.memories_recalled for t in turn_results[5:])
        self.assert_gte(
            "recall_active_mid_conversation", late_recalls, 1,
            f"Expected recall to return results in later turns, got {late_recalls} total recalls in turns 6-10",
        )

        recall_out = hebbs.recall(
            cue="engineering team knowledge management",
            strategy="similarity",
            top_k=10,
            entity_id=ENTITY,
        )
        self.assert_not_empty(
            "similarity_recall_returns_results", recall_out.results,
        )

        memories_with_context = 0
        for r in recall_out.results:
            if r.memory.context and len(r.memory.context) > 0:
                memories_with_context += 1

        self.assert_gte(
            "memories_have_context_metadata", memories_with_context, 1,
            f"Expected at least 1 memory with context metadata, got {memories_with_context}",
        )

        for r in recall_out.results:
            self.assert_true(
                "memory_has_entity_id",
                r.memory.entity_id == ENTITY,
                f"Memory entity_id should be '{ENTITY}', got '{r.memory.entity_id}'",
            )
            break

        self.assert_true(
            "no_strategy_errors",
            len(recall_out.strategy_errors) == 0,
            f"Recall had strategy errors: {recall_out.strategy_errors}",
        )

        count = hebbs.count()
        self.assert_gte(
            "total_memory_count", count, 5,
            f"HEBBS should have at least 5 memories after the call, got {count}",
        )
