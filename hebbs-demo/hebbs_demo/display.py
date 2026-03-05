"""DisplayManager: collapsible Rich panels showing HEBBS activity.

Every HEBBS operation renders a Rich panel with:
  - Always-visible title bar: operation name, count, latency
  - Collapsible detail section: memories, scores, edges, context
  - Color coding: insights (cyan), subscribe (yellow), latency (green/yellow/red)

Verbosity levels:
  --quiet:   no panels
  --normal:  collapsed (title bar only)
  --verbose: expanded (full details)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table


class Verbosity(Enum):
    QUIET = "quiet"
    NORMAL = "normal"
    VERBOSE = "verbose"


def _latency_color(ms: float) -> str:
    if ms < 10:
        return "green"
    elif ms < 100:
        return "yellow"
    return "red"


def _kind_badge(kind: str) -> Text:
    kind_upper = kind.upper() if kind else "EPISODE"
    color_map = {"EPISODE": "white", "INSIGHT": "cyan", "REVISION": "magenta"}
    color = color_map.get(kind_upper, "white")
    return Text(f"[{kind_upper}]", style=color)


@dataclass
class OperationRecord:
    """Captured data from a single HEBBS operation call."""
    operation: str
    latency_ms: float
    summary: str
    details: list[str] = field(default_factory=list)
    highlight_color: str = "white"
    embed_ms: float = 0.0
    llm_ms: float = 0.0


class DisplayManager:
    """Manages the terminal display of HEBBS operation activity."""

    def __init__(self, verbosity: Verbosity = Verbosity.NORMAL, console: Console | None = None) -> None:
        self.verbosity = verbosity
        self.console = console or Console()
        self._turn_records: list[OperationRecord] = []

    def start_turn(self) -> None:
        self._turn_records = []

    def record_operation(self, record: OperationRecord) -> None:
        self._turn_records.append(record)

    def display_turn(self) -> None:
        if self.verbosity == Verbosity.QUIET:
            return

        if not self._turn_records:
            self._display_no_activity()
            return

        for record in self._turn_records:
            self._display_record(record)

        onnx_ms = sum(r.embed_ms for r in self._turn_records)
        llm_ms = sum(r.llm_ms for r in self._turn_records)
        total_ms = sum(r.latency_ms for r in self._turn_records)
        hebbs_raw_ms = max(0.0, total_ms - onnx_ms - llm_ms)

        summary = Text()
        summary.append(" " * 30)
        summary.append("ONNX: ", style="dim")
        summary.append(f"{onnx_ms:.1f}ms", style=_latency_color(onnx_ms))
        summary.append("  |  ", style="dim")
        summary.append("HEBBS: ", style="dim")
        summary.append(f"{hebbs_raw_ms:.1f}ms", style=_latency_color(hebbs_raw_ms))
        summary.append("  |  ", style="dim")
        summary.append("LLM: ", style="dim")
        summary.append(f"{llm_ms:.1f}ms", style=_latency_color(llm_ms))
        summary.append("  |  ", style="dim")
        summary.append("total: ", style="dim")
        summary.append(f"{total_ms:.1f}ms", style=_latency_color(total_ms))
        self.console.print(summary)
        self.console.print()

    def _display_no_activity(self) -> None:
        text = Text()
        text.append("[", style="dim")
        text.append("·", style="dim yellow")
        text.append("] ", style="dim")
        text.append("HEBBS", style="dim bold")
        text.append("      no memories recalled, no subscriptions fired", style="dim")
        total_ms = sum(r.latency_ms for r in self._turn_records)
        latency_str = f"  {total_ms:.1f}ms" if total_ms > 0 else ""
        text.append(latency_str, style="dim")
        self.console.print(text)
        self.console.print()

    def _display_record(self, record: OperationRecord) -> None:
        expanded = self.verbosity == Verbosity.VERBOSE
        toggle = "-" if expanded else "+"

        header = Text()
        header.append(f"[{toggle}] ", style="bold")
        header.append(f"{record.operation:<10}", style=f"bold {record.highlight_color}")
        header.append(f" {record.summary}", style="white")
        padding = max(1, 70 - len(record.summary) - len(record.operation) - 5)
        header.append(" " * padding)
        header.append(f"{record.latency_ms:.1f}ms", style=_latency_color(record.latency_ms))
        self.console.print(header)

        if expanded and record.details:
            for detail in record.details:
                detail_text = Text()
                detail_text.append("    │ ", style="dim")
                if detail.startswith("✅"):
                    detail_text.append(detail, style="green")
                elif detail.startswith("❌"):
                    detail_text.append(detail, style="red dim")
                elif "[Insight]" in detail:
                    detail_text.append(detail, style="cyan")
                elif "surfaced" in detail.lower() or "confidence" in detail.lower():
                    detail_text.append(detail, style="yellow")
                else:
                    detail_text.append(detail, style="white")
                self.console.print(detail_text)

    def display_agent_response(self, response: str) -> None:
        self.console.print()
        text = Text()
        text.append("Agent: ", style="bold green")
        text.append(response)
        self.console.print(text)
        self.console.print()

    def display_prospect_message(self, entity: str, message: str) -> None:
        text = Text()
        label = entity if entity else "Prospect"
        text.append(f"{label}: ", style="bold blue")
        text.append(message)
        self.console.print(text)

    def display_session_header(self, entity_id: str, session_num: int | None = None) -> None:
        title = f"Session with {entity_id}"
        if session_num is not None:
            title += f" (#{session_num})"
        self.console.rule(title, style="bold cyan")

    def display_prime(
        self,
        entity_id: str,
        total: int,
        temporal_count: int,
        similarity_count: int,
        latency_ms: float,
        details: list[str] | None = None,
    ) -> None:
        """Display the prime operation results at session start."""
        if self.verbosity == Verbosity.QUIET:
            return

        supplementary = max(0, total - temporal_count - similarity_count)
        summary = f"{total} memories loaded for entity {entity_id}"
        detail_lines = [
            f"temporal:      {temporal_count} memories (recent history)",
            f"insights:      {similarity_count} memories (from reflect)",
            f"supplementary: {supplementary} memories (similar entities)",
        ]
        if details:
            detail_lines.extend(details)

        record = OperationRecord(
            operation="PRIME",
            latency_ms=latency_ms,
            summary=summary,
            details=detail_lines,
            highlight_color="cyan",
        )
        self._display_record(record)

    def display_reflect(
        self,
        memories_processed: int,
        clusters_found: int,
        insights_created: int,
        latency_ms: float,
        insight_details: list[str] | None = None,
        rejected_details: list[str] | None = None,
    ) -> None:
        """Display the reflect pipeline results."""
        if self.verbosity == Verbosity.QUIET:
            return

        summary = (
            f"{memories_processed} memories → {clusters_found} clusters → "
            f"{insights_created} insights"
        )
        details = []
        if insight_details:
            details.append("New insights:")
            details.extend(insight_details)
        if rejected_details:
            details.extend(rejected_details)

        record = OperationRecord(
            operation="REFLECT",
            latency_ms=latency_ms,
            summary=summary,
            details=details,
            highlight_color="magenta",
        )
        self._display_record(record)

    def display_forget(
        self,
        entity_id: str,
        forgotten_count: int,
        cascade_count: int,
        tombstone_count: int,
        latency_ms: float,
    ) -> None:
        """Display forget operation results."""
        if self.verbosity == Verbosity.QUIET:
            return

        summary = f"{forgotten_count} memories removed for entity {entity_id}"
        details = [
            f"memories:   {forgotten_count} deleted from all indexes",
            f"cascaded:   {cascade_count} predecessor snapshots deleted",
            f"tombstones: {tombstone_count} created (SHA-256 hashed)",
        ]
        record = OperationRecord(
            operation="FORGET",
            latency_ms=latency_ms,
            summary=summary,
            details=details,
            highlight_color="red",
        )
        self._display_record(record)

    def display_insights(
        self,
        insights: list[Any],
    ) -> None:
        """Display active insights at session start."""
        if self.verbosity == Verbosity.QUIET or not insights:
            return

        summary = f"{len(insights)} active insights for this scope"
        details = []
        for ins in insights[:10]:
            content = ins.content if hasattr(ins, "content") else str(ins)
            importance = ins.importance if hasattr(ins, "importance") else 0.0
            details.append(f'"{content[:60]}" ({importance:.2f}) [Insight]')

        record = OperationRecord(
            operation="INSIGHTS",
            latency_ms=0.0,
            summary=summary,
            details=details,
            highlight_color="cyan",
        )
        self._display_record(record)


class TimedOperation:
    """Context manager that measures wall-clock time for a HEBBS operation."""

    def __init__(self) -> None:
        self.start_ns: int = 0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> TimedOperation:
        self.start_ns = time.perf_counter_ns()
        return self

    def __exit__(self, *args: Any) -> None:
        self.elapsed_ms = (time.perf_counter_ns() - self.start_ns) / 1_000_000
