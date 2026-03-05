"""Unit tests for the display manager."""

from __future__ import annotations

from rich.console import Console

from hebbs_demo.display import (
    DisplayManager,
    OperationRecord,
    TimedOperation,
    Verbosity,
)


class TestOperationRecord:
    def test_basic(self):
        r = OperationRecord(
            operation="REMEMBER",
            latency_ms=3.2,
            summary="1 memory stored",
        )
        assert r.operation == "REMEMBER"
        assert r.latency_ms == 3.2

    def test_with_details(self):
        r = OperationRecord(
            operation="RECALL",
            latency_ms=4.1,
            summary="5 memories retrieved",
            details=[
                '0.91 "Beta Inc had identical SOC 2 blockers" [Episode]',
                '0.83 "Fintech prospects prioritize compliance" [Insight]',
            ],
        )
        assert len(r.details) == 2


class TestDisplayManager:
    def test_quiet_mode_no_output(self, capsys):
        console = Console(quiet=True)
        dm = DisplayManager(Verbosity.QUIET, console)
        dm.start_turn()
        dm.record_operation(OperationRecord("REMEMBER", 1.0, "1 stored"))
        dm.display_turn()

    def test_normal_mode_shows_headers(self):
        console = Console(file=None, force_terminal=True)
        dm = DisplayManager(Verbosity.NORMAL, console)
        dm.start_turn()
        dm.record_operation(OperationRecord("REMEMBER", 3.2, "1 stored"))
        dm.display_turn()

    def test_verbose_mode_shows_details(self):
        console = Console(file=None, force_terminal=True)
        dm = DisplayManager(Verbosity.VERBOSE, console)
        dm.start_turn()
        dm.record_operation(OperationRecord(
            "RECALL", 4.1, "5 retrieved",
            details=["0.91 memory 1 [Episode]", "0.83 memory 2 [Insight]"],
        ))
        dm.display_turn()

    def test_no_activity_display(self):
        console = Console(file=None, force_terminal=True)
        dm = DisplayManager(Verbosity.NORMAL, console)
        dm.start_turn()
        dm.display_turn()

    def test_multiple_records(self):
        console = Console(file=None, force_terminal=True)
        dm = DisplayManager(Verbosity.NORMAL, console)
        dm.start_turn()
        dm.record_operation(OperationRecord("REMEMBER", 3.2, "1 stored"))
        dm.record_operation(OperationRecord("RECALL", 4.1, "5 retrieved"))
        dm.record_operation(OperationRecord("SUBSCRIBE", 0.8, "1 surfaced", highlight_color="yellow"))
        dm.display_turn()

    def test_display_prime(self):
        console = Console(file=None, force_terminal=True)
        dm = DisplayManager(Verbosity.VERBOSE, console)
        dm.display_prime("acme_corp", 14, 8, 4, 6.2)

    def test_display_forget(self):
        console = Console(file=None, force_terminal=True)
        dm = DisplayManager(Verbosity.VERBOSE, console)
        dm.display_forget("acme_corp", 20, 3, 20, 12.1)


class TestTimedOperation:
    def test_measures_time(self):
        import time
        with TimedOperation() as t:
            time.sleep(0.01)
        assert t.elapsed_ms >= 5  # at least 5ms
        assert t.elapsed_ms < 500  # sanity upper bound
