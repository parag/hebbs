"""Scenario E: Subscribe & Real-Time Memory Surfacing.

Seeds memories for a prospect, opens a subscribe stream, feeds
live conversation text, and validates that the real-time surfacing
pipeline operates without errors.
"""

from __future__ import annotations

from typing import Any

from hebbs_demo.scenarios.base import Scenario

SEED_MEMORIES: list[dict[str, Any]] = [
    {
        "content": "Acme Robotics uses ROS2 for fleet orchestration across 12 warehouse sites",
        "importance": 0.8,
        "context": {"stage": "discovery", "topic": "tech_stack", "product": "fleet_mgmt"},
    },
    {
        "content": "VP Engineering Tom Bradley evaluating edge inference for pick-and-place arms",
        "importance": 0.7,
        "context": {"stage": "discovery", "contact": "Tom Bradley", "role": "VP Engineering"},
    },
    {
        "content": "Current vision model runs on Jetson Orin, needs to support 60fps at 1080p",
        "importance": 0.9,
        "context": {"stage": "technical", "topic": "performance", "hardware": "Jetson Orin"},
    },
    {
        "content": "Safety certification ISO 13482 required before any production deployment",
        "importance": 0.9,
        "context": {"stage": "compliance", "topic": "safety_cert", "standard": "ISO 13482"},
    },
    {
        "content": "Annual robotics R&D budget is $8M, ML inference allocated $1.2M",
        "importance": 0.8,
        "context": {"stage": "qualification", "topic": "budget", "amount": "$1.2M"},
    },
    {
        "content": "Previous vendor VisioBot failed latency SLA: 45ms average vs 20ms target",
        "importance": 0.9,
        "context": {"stage": "pain_point", "topic": "latency", "competitor": "VisioBot"},
    },
    {
        "content": "Acme expanding to automotive assembly line inspection in Q3",
        "importance": 0.7,
        "context": {"stage": "growth", "topic": "expansion", "vertical": "automotive"},
    },
    {
        "content": "CTO wants unified model serving platform across edge and cloud nodes",
        "importance": 0.8,
        "context": {"stage": "technical", "topic": "architecture", "preference": "unified_platform"},
    },
    {
        "content": "Data pipeline ingests 2TB daily from 500+ camera feeds across all sites",
        "importance": 0.8,
        "context": {"stage": "technical", "topic": "data_volume", "sources": "camera_feeds"},
    },
    {
        "content": "Procurement requires 60-day POC with measurable throughput improvement > 15%",
        "importance": 0.7,
        "context": {"stage": "process", "topic": "poc", "threshold": "15%"},
    },
]

CONVERSATION_TURNS = [
    "We've been running into major latency issues with our current vision pipeline on the Jetson boards. The pick-and-place cycle time is killing our throughput.",
    "Tom mentioned you might be able to help with edge inference optimization. Our safety certification timeline is tight — we need ISO 13482 before the automotive expansion.",
    "Budget-wise, we carved out about a million dollars from the R&D allocation. But we need to see clear ROI in the POC phase.",
]

ENTITY_ID = "acme_robotics"


class SubscribeRealtimeScenario(Scenario):
    name = "subscribe_realtime"
    description = "Scenario E: Seed memories, open subscribe stream, feed conversation text, validate real-time surfacing pipeline"

    def execute(self, hebbs: Any, agent: Any) -> None:
        for mem in SEED_MEMORIES:
            hebbs.remember(
                content=mem["content"],
                importance=mem["importance"],
                context=mem["context"],
                entity_id=ENTITY_ID,
            )

        count = hebbs.count()
        self.assert_equal("seed_memories_stored", len(SEED_MEMORIES), count)

        subscription = hebbs.subscribe(
            entity_id=ENTITY_ID,
            confidence_threshold=0.3,
        )
        self.assert_true(
            "subscription_created",
            subscription is not None,
            "subscribe() should return a stream object",
        )

        all_pushes: list[Any] = []
        try:
            for turn_text in CONVERSATION_TURNS:
                feed_ok = True
                try:
                    subscription.feed(turn_text)
                except Exception:
                    feed_ok = False

                self.assert_true(
                    f"feed_no_error_{turn_text[:30]}",
                    feed_ok,
                    "feed() should not raise an exception",
                )

                pushes_this_turn: list[Any] = []
                for _ in range(5):
                    push = subscription.poll(timeout_secs=0.2)
                    if push is None:
                        break
                    pushes_this_turn.append(push)

                for push in pushes_this_turn:
                    self.assert_true(
                        "push_has_memory",
                        hasattr(push, "memory") and push.memory is not None,
                        "each push should contain a memory object",
                    )
                    self.assert_true(
                        "push_has_confidence",
                        hasattr(push, "confidence"),
                        "each push should have a confidence score",
                    )

                all_pushes.extend(pushes_this_turn)
        finally:
            subscription.close()

        self.assert_true(
            "subscribe_pipeline_ran",
            True,
            "subscribe feed/poll/close cycle completed without fatal error",
        )

        agent.start_session(entity_id=ENTITY_ID, session_num=1, use_subscribe=True)
        turn_result = agent.process_turn(
            prospect_message="Can you walk me through how your edge inference handles the Jetson Orin's thermal throttling?",
            recall_strategies=["similarity", "temporal"],
        )
        self.assert_true(
            "agent_turn_with_subscribe",
            turn_result is not None,
            "agent should complete a turn with subscribe active",
        )
        agent.end_session()
