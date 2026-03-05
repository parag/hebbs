"""Async API for HEBBS (server mode only).

Usage:
    from hebbs.aio import HEBBS

    async def main():
        h = await HEBBS.connect("localhost:6380")
        memory = await h.remember("customer prefers email", importance=0.8)
        results = await h.recall("contact preferences")

Note: Async is only available for server mode. Embedded mode
operations are synchronous with GIL released during Rust computation.
"""

from __future__ import annotations

from hebbs._exceptions import ConfigurationError


class HEBBS:
    """Async HEBBS client for server mode.

    Note: This is a placeholder. Full async implementation requires
    grpcio.aio which will be implemented when server mode is enabled.
    """

    @classmethod
    async def connect(cls, endpoint: str = "localhost:6380", *, timeout: float = 30.0) -> "HEBBS":
        raise ConfigurationError(
            "Async server mode is not yet implemented. "
            "Use embedded mode: from hebbs import HEBBS; h = HEBBS.open('./path')"
        )
