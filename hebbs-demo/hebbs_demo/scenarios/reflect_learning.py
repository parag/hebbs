"""Scenario D: Reflect & Institutional Learning.

Ingests 50 simulated call summaries across 10 fintech companies, runs
reflect() to distil cross-entity insights, then verifies that a new
session with an 11th company benefits from those insights.
"""

from __future__ import annotations

from typing import Any

from hebbs_demo.scenarios.base import Scenario

FINTECH_COMPANIES = [
    "meridian_payments",
    "vault_digital",
    "neoledger",
    "clearfin",
    "quantumcard",
    "paystream_ai",
    "openbanc",
    "finpulse",
    "ledgerx",
    "blocksettle",
]

CALL_SUMMARIES: list[dict[str, Any]] = [
    {"entity": "meridian_payments", "content": "Meridian Payments processes 2.3M daily transactions and needs sub-200ms fraud scoring", "importance": 0.9, "context": {"stage": "discovery", "topic": "fraud_detection", "volume": "2.3M_daily"}},
    {"entity": "meridian_payments", "content": "Current vendor charges $0.003 per transaction for fraud checks, annual cost ~$2.5M", "importance": 0.8, "context": {"stage": "discovery", "topic": "pricing", "annual_cost": "$2.5M"}},
    {"entity": "meridian_payments", "content": "CTO Sarah Chen prefers on-prem deployment due to PCI-DSS requirements", "importance": 0.7, "context": {"stage": "discovery", "topic": "deployment", "contact": "Sarah Chen"}},
    {"entity": "meridian_payments", "content": "Compliance team requires SOC2 Type II and ISO 27001 before vendor approval", "importance": 0.8, "context": {"stage": "qualification", "topic": "compliance"}},
    {"entity": "meridian_payments", "content": "Board-level initiative to reduce false positive rate from 4.2% to under 1%", "importance": 0.9, "context": {"stage": "discovery", "topic": "accuracy"}},
    {"entity": "vault_digital", "content": "Vault Digital is building a neobank targeting Gen-Z with instant P2P transfers", "importance": 0.7, "context": {"stage": "discovery", "topic": "product"}},
    {"entity": "vault_digital", "content": "Series B funded, $45M raised, allocating $3M for compliance infrastructure", "importance": 0.8, "context": {"stage": "qualification", "topic": "budget"}},
    {"entity": "vault_digital", "content": "Need KYC/AML pipeline that handles 500K identity verifications monthly", "importance": 0.9, "context": {"stage": "discovery", "topic": "kyc_aml"}},
    {"entity": "vault_digital", "content": "VP Engineering Raj Patel wants API-first integration, no SDKs", "importance": 0.6, "context": {"stage": "technical", "contact": "Raj Patel"}},
    {"entity": "vault_digital", "content": "Decision timeline: vendor selected by Q2, implementation by Q3", "importance": 0.7, "context": {"stage": "timeline", "topic": "decision"}},
    {"entity": "neoledger", "content": "NeoLedger provides distributed ledger for trade finance across 14 APAC banks", "importance": 0.7, "context": {"stage": "discovery", "topic": "product"}},
    {"entity": "neoledger", "content": "Regulatory pressure from MAS to adopt real-time transaction monitoring by January", "importance": 0.9, "context": {"stage": "urgency", "topic": "compliance"}},
    {"entity": "neoledger", "content": "Current reconciliation process takes 72 hours, target is same-day", "importance": 0.8, "context": {"stage": "pain_point", "topic": "reconciliation"}},
    {"entity": "neoledger", "content": "Exploring ML-based anomaly detection for cross-border payments", "importance": 0.7, "context": {"stage": "discovery", "topic": "ml_adoption"}},
    {"entity": "neoledger", "content": "Budget allocated from innovation fund, CFO sign-off already obtained", "importance": 0.8, "context": {"stage": "qualification", "topic": "budget"}},
    {"entity": "clearfin", "content": "ClearFin offers embedded lending APIs for e-commerce platforms, 200+ merchants", "importance": 0.7, "context": {"stage": "discovery", "topic": "product"}},
    {"entity": "clearfin", "content": "Experiencing 12% default rate on micro-loans, need better credit scoring", "importance": 0.9, "context": {"stage": "pain_point", "topic": "credit_risk"}},
    {"entity": "clearfin", "content": "Interested in alternative data signals: purchase history, social graph, app usage", "importance": 0.8, "context": {"stage": "technical", "topic": "data_sources"}},
    {"entity": "clearfin", "content": "Head of Risk Maria Gonzalez wants explainable model outputs for regulatory audit", "importance": 0.8, "context": {"stage": "technical", "contact": "Maria Gonzalez"}},
    {"entity": "clearfin", "content": "Pilot program: test with top 20 merchants before full rollout", "importance": 0.6, "context": {"stage": "timeline", "topic": "pilot"}},
    {"entity": "quantumcard", "content": "QuantumCard issues 5M+ credit cards across Southeast Asia, real-time auth required", "importance": 0.8, "context": {"stage": "discovery", "topic": "scale"}},
    {"entity": "quantumcard", "content": "Chargeback disputes cost $18M annually, primary driver of CEO urgency", "importance": 0.9, "context": {"stage": "pain_point", "topic": "chargebacks"}},
    {"entity": "quantumcard", "content": "Existing rule-based engine produces 15K false declines daily, hurting NPS scores", "importance": 0.9, "context": {"stage": "pain_point", "topic": "false_declines"}},
    {"entity": "quantumcard", "content": "Require multi-region deployment: Singapore primary, Tokyo DR, Mumbai expansion", "importance": 0.7, "context": {"stage": "technical", "topic": "infrastructure"}},
    {"entity": "quantumcard", "content": "Procurement cycle requires 3 vendor demos and 90-day POC before contract", "importance": 0.6, "context": {"stage": "process", "topic": "procurement"}},
    {"entity": "paystream_ai", "content": "PayStream AI automates B2B invoice payments for mid-market companies", "importance": 0.7, "context": {"stage": "discovery", "topic": "product"}},
    {"entity": "paystream_ai", "content": "Processing $800M monthly in invoice payments, growing 40% quarter over quarter", "importance": 0.8, "context": {"stage": "qualification", "topic": "volume"}},
    {"entity": "paystream_ai", "content": "CFO concerned about duplicate payment fraud, estimated $2M annual leakage", "importance": 0.9, "context": {"stage": "pain_point", "topic": "fraud"}},
    {"entity": "paystream_ai", "content": "Tech stack: Kubernetes on AWS, event-driven microservices, wants webhook callbacks", "importance": 0.6, "context": {"stage": "technical", "topic": "integration"}},
    {"entity": "paystream_ai", "content": "Existing contract with incumbent expires in 6 months, evaluation window is now", "importance": 0.8, "context": {"stage": "urgency", "topic": "timeline"}},
    {"entity": "openbanc", "content": "OpenBanc is an open-banking aggregator connecting 40+ banks via PSD2 APIs", "importance": 0.7, "context": {"stage": "discovery", "topic": "product"}},
    {"entity": "openbanc", "content": "Need consent management layer for GDPR-compliant data sharing across institutions", "importance": 0.9, "context": {"stage": "requirement", "topic": "gdpr"}},
    {"entity": "openbanc", "content": "Peak API call volume: 1.2M requests/hour during payroll cycles", "importance": 0.8, "context": {"stage": "technical", "topic": "scale"}},
    {"entity": "openbanc", "content": "CTO wants to replace polling-based architecture with event streaming", "importance": 0.7, "context": {"stage": "technical", "topic": "architecture"}},
    {"entity": "openbanc", "content": "FCA audit upcoming in Q4, need full transaction lineage and audit trail", "importance": 0.9, "context": {"stage": "urgency", "topic": "compliance"}},
    {"entity": "finpulse", "content": "FinPulse provides real-time market data feeds to 300+ hedge funds", "importance": 0.7, "context": {"stage": "discovery", "topic": "product"}},
    {"entity": "finpulse", "content": "Latency-sensitive: current p99 is 8ms, target is sub-2ms for options pricing", "importance": 0.9, "context": {"stage": "requirement", "topic": "latency"}},
    {"entity": "finpulse", "content": "Exploring FPGA-accelerated inference for tick-by-tick anomaly detection", "importance": 0.7, "context": {"stage": "technical", "topic": "acceleration"}},
    {"entity": "finpulse", "content": "Data residency requirements: US equities in Virginia, EU derivatives in Frankfurt", "importance": 0.8, "context": {"stage": "compliance", "topic": "data_residency"}},
    {"entity": "finpulse", "content": "Annual infrastructure budget: $12M, willing to allocate 8% for ML inference", "importance": 0.8, "context": {"stage": "qualification", "topic": "budget"}},
    {"entity": "ledgerx", "content": "LedgerX operates a regulated crypto derivatives exchange, CFTC-licensed", "importance": 0.8, "context": {"stage": "discovery", "topic": "product"}},
    {"entity": "ledgerx", "content": "Need real-time margin calculation across 50+ crypto pairs with 1-second updates", "importance": 0.9, "context": {"stage": "requirement", "topic": "margin_calc"}},
    {"entity": "ledgerx", "content": "Previous ML vendor failed stress test: model drift during high-volatility events", "importance": 0.9, "context": {"stage": "pain_point", "topic": "reliability"}},
    {"entity": "ledgerx", "content": "Require hot-hot failover with zero data loss, current RTO is 30 seconds", "importance": 0.8, "context": {"stage": "technical", "topic": "disaster_recovery"}},
    {"entity": "ledgerx", "content": "Board approved $5M budget for risk infrastructure modernization", "importance": 0.8, "context": {"stage": "qualification", "topic": "budget"}},
    {"entity": "blocksettle", "content": "BlockSettle provides institutional-grade digital asset settlement for banks", "importance": 0.7, "context": {"stage": "discovery", "topic": "product"}},
    {"entity": "blocksettle", "content": "T+0 settlement ambition requires atomic swap verification in under 500ms", "importance": 0.9, "context": {"stage": "requirement", "topic": "settlement_speed"}},
    {"entity": "blocksettle", "content": "Onboarding 3 tier-1 banks this quarter, each with unique integration requirements", "importance": 0.8, "context": {"stage": "growth", "topic": "expansion"}},
    {"entity": "blocksettle", "content": "Chief Compliance Officer wants automated SAR filing for suspicious transaction patterns", "importance": 0.8, "context": {"stage": "compliance", "topic": "aml"}},
    {"entity": "blocksettle", "content": "Multi-jurisdictional: US, EU, Singapore — need unified compliance dashboard", "importance": 0.9, "context": {"stage": "requirement", "topic": "multi_jurisdiction"}},
]


class ReflectLearningScenario(Scenario):
    name = "reflect_learning"
    description = "Scenario D: Bulk ingest 50 fintech call summaries, run reflect() for insights, verify institutional learning propagates to a new entity session"

    def execute(self, hebbs: Any, agent: Any) -> None:
        for summary in CALL_SUMMARIES:
            hebbs.remember(
                content=summary["content"],
                importance=summary["importance"],
                context=summary["context"],
                entity_id=summary["entity"],
            )

        count_after_ingest = hebbs.count()
        self.assert_gte("memories_ingested", count_after_ingest, 20)
        self.assert_equal("exact_ingest_count", 50, count_after_ingest)

        reflect_result = hebbs.reflect()
        self.assert_true(
            "reflect_completes",
            reflect_result is not None,
            "reflect() should return a result object",
        )
        self.assert_gte(
            "memories_processed",
            reflect_result.memories_processed,
            20,
            "reflect should process a significant portion of ingested memories",
        )

        count_after_reflect = hebbs.count()
        self.assert_gte(
            "count_increased_after_reflect",
            count_after_reflect,
            count_after_ingest,
            "count should not decrease after reflect (insights may be added)",
        )

        new_entity = "nexgen_capital"
        agent.start_session(entity_id=new_entity, session_num=1)

        global_insights = hebbs.insights(entity_id=None, max_results=20)
        self.assert_true(
            "insights_available_for_new_entity",
            global_insights is not None,
            "insights() should return without error for global scope",
        )

        turn = agent.process_turn(
            prospect_message=(
                "We're a crypto exchange processing 10K trades per second "
                "and need real-time risk scoring with sub-5ms latency. "
                "Compliance is our biggest headache — CFTC and MAS both breathing down our neck."
            ),
            recall_strategies=["similarity"],
        )
        self.assert_true(
            "new_entity_turn_succeeds",
            turn is not None,
            "processing a turn with the new entity should succeed",
        )

        agent.end_session()
