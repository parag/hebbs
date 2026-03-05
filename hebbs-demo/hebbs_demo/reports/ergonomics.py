"""API ergonomics audit collector.

Collects friction points encountered during demo development
and generates the ERGONOMICS_REPORT.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Severity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Category(Enum):
    MISSING_CONVENIENCE = "Missing convenience methods"
    AWKWARD_TYPES = "Awkward type conversions"
    UNHELPFUL_ERRORS = "Unhelpful error messages"
    MISSING_FEATURES = "Missing features"
    PERFORMANCE_SURPRISES = "Performance surprises"
    DOCUMENTATION_GAPS = "Documentation gaps"


@dataclass
class FrictionPoint:
    category: Category
    severity: Severity
    title: str
    description: str
    suggestion: str


FINDINGS: list[FrictionPoint] = [
    FrictionPoint(
        category=Category.MISSING_CONVENIENCE,
        severity=Severity.MEDIUM,
        title="No shorthand for similarity recall",
        description=(
            "Every similarity recall requires constructing the full call: "
            "hebbs.recall(cue=text, strategy='similarity', top_k=10, entity_id=eid). "
            "A shorthand like hebbs.recall_similar(text, top_k=10) would cover 80% of use cases."
        ),
        suggestion="Add HEBBS.recall_similar(cue, top_k=10, entity_id=None) convenience method.",
    ),
    FrictionPoint(
        category=Category.MISSING_CONVENIENCE,
        severity=Severity.LOW,
        title="No batch remember()",
        description=(
            "Ingesting 50 call summaries requires 50 individual remember() calls. "
            "A batch API would reduce overhead for bulk ingestion scenarios."
        ),
        suggestion="Add HEBBS.remember_batch(items: list[dict]) -> list[Memory].",
    ),
    FrictionPoint(
        category=Category.MISSING_FEATURES,
        severity=Severity.HIGH,
        title="No custom embedder in embedded mode",
        description=(
            "HEBBS.open() only supports mock or ONNX embedder. There is no way to pass "
            "a custom embedder (e.g., OpenAI embeddings) through the Python API in embedded mode. "
            "This limits embedding provider comparison to mock vs ONNX."
        ),
        suggestion=(
            "Add an embedder parameter to HEBBS.open() or support a callback-based embedder "
            "that delegates to Python-side embedding code."
        ),
    ),
    FrictionPoint(
        category=Category.MISSING_FEATURES,
        severity=Severity.HIGH,
        title="No LLM provider configuration in embedded mode",
        description=(
            "The reflect pipeline in embedded mode uses the default (mock) LLM provider. "
            "There is no API to configure OpenAI/Anthropic/Ollama as the reflect LLM in embedded mode. "
            "This means reflect() in the demo produces mock insights, not real LLM-generated ones."
        ),
        suggestion=(
            "Add llm_provider parameter to HEBBS.open() or a set_llm_provider() method, "
            "or allow configuring LLM via environment variables."
        ),
    ),
    FrictionPoint(
        category=Category.AWKWARD_TYPES,
        severity=Severity.MEDIUM,
        title="RecallStrategy accepts string or enum inconsistently",
        description=(
            "recall(strategy='similarity') and recall(strategy=RecallStrategy.SIMILARITY) both work, "
            "but the string values are not documented in the type hints. Users discover valid strings "
            "by reading source code."
        ),
        suggestion="Document valid string values in the docstring. Consider a Literal type hint.",
    ),
    FrictionPoint(
        category=Category.UNHELPFUL_ERRORS,
        severity=Severity.MEDIUM,
        title="Generic InvalidInputError for multiple failure modes",
        description=(
            "When recall() fails due to missing entity_id for temporal strategy, the error says "
            "'invalid input' without specifying which parameter is wrong or what strategy requires it."
        ),
        suggestion="Include the strategy name and the missing parameter in the error message.",
    ),
    FrictionPoint(
        category=Category.PERFORMANCE_SURPRISES,
        severity=Severity.LOW,
        title="First remember() with ONNX embedder is slow (model download)",
        description=(
            "The first remember() call with use_mock_embedder=False can take 5-30 seconds "
            "as it downloads the ONNX model. There is no progress indicator or warning."
        ),
        suggestion="Add a warmup() method or print a message during model download.",
    ),
    FrictionPoint(
        category=Category.DOCUMENTATION_GAPS,
        severity=Severity.MEDIUM,
        title="Unclear which context fields are meaningful",
        description=(
            "The context parameter accepts any dict, but it's unclear which keys HEBBS uses "
            "internally (e.g., for analogical recall's structural similarity). Users don't know "
            "whether their context keys affect recall quality."
        ),
        suggestion="Document recommended context keys and explain how they affect each recall strategy.",
    ),
    FrictionPoint(
        category=Category.MISSING_FEATURES,
        severity=Severity.MEDIUM,
        title="No way to list all memories for an entity",
        description=(
            "There's no hebbs.list(entity_id=X) method. To see all memories for an entity, "
            "you must use recall() with a broad cue, which is imprecise."
        ),
        suggestion="Add HEBBS.list(entity_id=None, limit=100) -> list[Memory].",
    ),
    FrictionPoint(
        category=Category.MISSING_CONVENIENCE,
        severity=Severity.LOW,
        title="No way to get subscribe stats",
        description=(
            "The SubscribeStream wrapper doesn't expose the underlying stats (chunks_processed, "
            "memories_pushed, bloom_rejections). This makes it hard to debug subscribe behavior."
        ),
        suggestion="Add SubscribeStream.stats() -> dict method wrapping the native stats.",
    ),
]


def generate_report() -> str:
    """Generate the ERGONOMICS_REPORT.md content."""
    lines = [
        "# API Ergonomics Report",
        "",
        "Friction points discovered during Phase 14 reference application development.",
        "",
        "---",
        "",
    ]

    by_category: dict[Category, list[FrictionPoint]] = {}
    for fp in FINDINGS:
        by_category.setdefault(fp.category, []).append(fp)

    severity_order = {Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.MEDIUM: 2, Severity.LOW: 3}

    for category in Category:
        fps = by_category.get(category, [])
        if not fps:
            continue

        fps.sort(key=lambda f: severity_order[f.severity])

        lines.append(f"## {category.value}")
        lines.append("")

        for fp in fps:
            sev = fp.severity.value.upper()
            lines.append(f"### [{sev}] {fp.title}")
            lines.append("")
            lines.append(fp.description)
            lines.append("")
            lines.append(f"**Suggestion:** {fp.suggestion}")
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Severity | Count |")
    lines.append("|----------|-------|")
    for sev in Severity:
        count = sum(1 for fp in FINDINGS if fp.severity == sev)
        if count:
            lines.append(f"| {sev.value.title()} | {count} |")
    lines.append(f"| **Total** | **{len(FINDINGS)}** |")
    lines.append("")

    return "\n".join(lines)
