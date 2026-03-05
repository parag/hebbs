"""LLM and embedding provider comparison report generators.

Runs identical scenarios across configured providers and produces
structured comparison reports with latency, quality, and cost metrics.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from hebbs_demo.config import DemoConfig
from hebbs_demo.display import DisplayManager, Verbosity


@dataclass
class ProviderMetrics:
    provider: str
    model: str
    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    extraction_parse_success: int = 0
    extraction_total: int = 0
    memories_created: int = 0
    errors: int = 0

    @property
    def p50_latency(self) -> float:
        if not self.latencies_ms:
            return 0
        s = sorted(self.latencies_ms)
        return s[len(s) // 2]

    @property
    def p99_latency(self) -> float:
        if not self.latencies_ms:
            return 0
        s = sorted(self.latencies_ms)
        idx = min(int(len(s) * 0.99), len(s) - 1)
        return s[idx]

    @property
    def estimated_cost_usd(self) -> float:
        return (self.total_input_tokens * 2.50 + self.total_output_tokens * 10.00) / 1_000_000

    @property
    def extraction_success_rate(self) -> float:
        if self.extraction_total == 0:
            return 0
        return self.extraction_parse_success / self.extraction_total


def _run_scenario_for_provider(
    provider_name: str,
    model: str,
    config: DemoConfig,
    verbosity: Verbosity,
) -> ProviderMetrics:
    """Run the reflect learning scenario and collect metrics for a provider."""
    from hebbs_demo.scenarios.reflect_learning import ReflectLearningScenario

    metrics = ProviderMetrics(provider=provider_name, model=model)

    override_config = DemoConfig.from_toml(Path("configs/openai.toml")) if Path("configs/openai.toml").exists() else config
    override_config.llm.conversation_provider = provider_name
    override_config.llm.conversation_model = model
    override_config.llm.extraction_provider = provider_name
    override_config.llm.extraction_model = model

    scenario = ReflectLearningScenario(
        config=override_config,
        verbosity=verbosity,
        use_mock_llm=True,
        console=Console(quiet=True),
    )

    t0 = time.perf_counter()
    result = scenario.run()
    elapsed = (time.perf_counter() - t0) * 1000

    metrics.total_calls = 1
    metrics.latencies_ms = [elapsed]
    metrics.memories_created = sum(1 for a in result.assertions if a.passed)

    if not result.passed:
        metrics.errors = len(result.failed_assertions)

    return metrics


def run_llm_comparison(config: DemoConfig, verbosity: Verbosity, console: Console) -> None:
    """Compare LLM providers by running the same scenario with each."""
    console.print("\n[bold]LLM Provider Comparison[/bold]")
    console.print("[dim]Running reflect_learning scenario with mock LLM for each provider...[/dim]\n")

    providers = [
        ("openai", config.llm.openai.model or "gpt-4o"),
        ("anthropic", config.llm.anthropic.model or "claude-sonnet-4-20250514"),
        ("ollama", config.llm.ollama.model or "llama3.2"),
    ]

    all_metrics: list[ProviderMetrics] = []
    for prov_name, model in providers:
        console.print(f"  Running with {prov_name}/{model}...", end=" ")
        try:
            metrics = _run_scenario_for_provider(prov_name, model, config, verbosity)
            all_metrics.append(metrics)
            console.print("[green]done[/green]")
        except Exception as e:
            console.print(f"[red]failed: {e}[/red]")
            all_metrics.append(ProviderMetrics(provider=prov_name, model=model, errors=1))

    _print_comparison_table(all_metrics, console)


def _print_comparison_table(metrics_list: list[ProviderMetrics], console: Console) -> None:
    table = Table(title="LLM Provider Comparison", show_header=True, header_style="bold")
    table.add_column("Provider", style="white")
    table.add_column("Model", style="cyan")
    table.add_column("p50 Latency", justify="right")
    table.add_column("p99 Latency", justify="right")
    table.add_column("Memories", justify="right")
    table.add_column("Errors", justify="right")
    table.add_column("Est. Cost", justify="right")

    for m in metrics_list:
        table.add_row(
            m.provider,
            m.model,
            f"{m.p50_latency:.0f}ms",
            f"{m.p99_latency:.0f}ms",
            str(m.memories_created),
            str(m.errors),
            f"${m.estimated_cost_usd:.4f}",
        )

    console.print()
    console.print(table)


def run_embedding_comparison(
    config_paths: list[str], verbosity: Verbosity, console: Console,
) -> None:
    """Compare embedding providers by running the same scenario with each."""
    console.print("\n[bold]Embedding Provider Comparison[/bold]")
    console.print("[dim]Running forget_gdpr scenario with each embedding config...[/dim]\n")

    from hebbs_demo.scenarios.forget_gdpr import ForgetGdprScenario

    results = []
    for path in config_paths:
        console.print(f"  Config: {path}...", end=" ")
        try:
            cfg = DemoConfig.from_toml(path)
            scenario = ForgetGdprScenario(
                config=cfg,
                verbosity=verbosity,
                use_mock_llm=True,
                console=Console(quiet=True),
            )
            t0 = time.perf_counter()
            result = scenario.run()
            elapsed = (time.perf_counter() - t0) * 1000
            results.append((path, cfg.embedding.provider, result, elapsed))
            status = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
            console.print(f"{status} ({elapsed:.0f}ms)")
        except Exception as e:
            console.print(f"[red]failed: {e}[/red]")

    if results:
        table = Table(title="Embedding Provider Comparison", show_header=True, header_style="bold")
        table.add_column("Config")
        table.add_column("Provider")
        table.add_column("Status")
        table.add_column("Assertions")
        table.add_column("Time", justify="right")

        for path, provider, result, elapsed in results:
            passed = sum(1 for a in result.assertions if a.passed)
            status = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
            table.add_row(
                Path(path).name, provider, status,
                f"{passed}/{len(result.assertions)}", f"{elapsed:.0f}ms",
            )

        console.print()
        console.print(table)
