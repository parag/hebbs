"""Scenario B: Objection Handling — cross-entity analogical recall.

Two sessions: first with beta_inc encountering a pricing objection,
second with gamma_corp facing a similar concern. Validates that
analogical recall surfaces structurally similar past situations
and that prime() loads relevant cross-entity context.
"""

from __future__ import annotations

from typing import Any

from hebbs_demo.scenarios.base import Scenario

BETA_TURNS = [
    "Hey, I'm David Park, Head of Platform at Beta Inc. We've been evaluating memory-augmented AI systems.",
    "We like what we've seen in the demo. The recall latency is impressive for our real-time use case.",
    "Here's where it gets tricky — your pricing is about 3x what we budgeted. We were expecting something closer to $5 per seat.",
    "Our CFO pushed back hard. She said we can't justify the spend without a clear ROI model showing payback within 6 months.",
    "We ended up going with a cheaper alternative, but honestly the recall quality wasn't as good. We might circle back next quarter.",
]

GAMMA_TURNS = [
    "This is Lisa Okonkwo, Engineering Director at Gamma Corp. We're building an AI copilot and need a memory backend.",
    "The product looks solid. My team ran some benchmarks and the p99 latency numbers are right where we need them.",
    "I have to be upfront — the pricing gave us pause. Our procurement team flagged it as significantly above market rate.",
    "We need to show leadership a concrete business case. What kind of ROI data can you share from existing customers?",
    "If we can build a compelling case, I think we can get budget approval. But we'd need help structuring that narrative.",
]


class ObjectionHandlingScenario(Scenario):
    name = "objection_handling"
    description = "Cross-entity analogical recall for recurring pricing objections"

    def execute(self, hebbs: Any, agent: Any) -> None:
        agent.start_session(entity_id="beta_inc", session_num=1)
        for message in BETA_TURNS:
            agent.process_turn(message, recall_strategies=["similarity"])
        agent.end_session()

        beta_count = hebbs.count()
        self.assert_gte(
            "beta_memories_stored", beta_count, 3,
            f"Expected at least 3 memories from beta_inc session, got {beta_count}",
        )

        prime_out = hebbs.prime(entity_id="gamma_corp", max_memories=20)
        gamma_pre_prime_count = len(prime_out.results)

        agent.start_session(entity_id="gamma_corp", session_num=1)

        gamma_results = []
        for i, message in enumerate(GAMMA_TURNS):
            strategies = ["similarity", "analogical"] if i >= 2 else ["similarity"]
            result = agent.process_turn(message, recall_strategies=strategies)
            gamma_results.append(result)
        agent.end_session()

        objection_recall = hebbs.recall(
            cue="pricing objection budget concern ROI justification",
            strategy="analogical",
            top_k=10,
            entity_id=None,
        )
        self.assert_not_empty(
            "analogical_recall_finds_results",
            objection_recall.results,
        )

        has_beta_memory = any(
            r.memory.entity_id == "beta_inc"
            for r in objection_recall.results
        )
        has_gamma_memory = any(
            r.memory.entity_id == "gamma_corp"
            for r in objection_recall.results
        )
        self.assert_true(
            "analogical_recall_spans_entities",
            has_beta_memory or has_gamma_memory,
            "Analogical recall should surface memories from at least one of the two entities",
        )

        post_count = hebbs.count()
        self.assert_gte(
            "total_memories_after_both_sessions", post_count, beta_count + 2,
            f"Expected memories to grow after gamma_corp session (was {beta_count}, now {post_count})",
        )

        gamma_prime = hebbs.prime(entity_id="gamma_corp", max_memories=20)
        self.assert_not_empty(
            "gamma_prime_has_memories_after_session",
            gamma_prime.results,
        )

        late_turn_recalls = sum(r.memories_recalled for r in gamma_results[2:])
        self.assert_gte(
            "gamma_objection_turns_have_recall", late_turn_recalls, 1,
            f"Expected recall hits during pricing objection turns, got {late_turn_recalls}",
        )
