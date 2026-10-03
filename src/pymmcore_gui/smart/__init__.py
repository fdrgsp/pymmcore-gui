"""Public API for Smart Microscopy (event-driven acquisition) analysis scripts.

A script is a single ``.py`` file defining ``analyze(image, frame, ctx)``,
optionally ``setup(ctx)`` / ``teardown(ctx)``, and a few literal constants
(``API_VERSION``, ``NAME``, ``PARAMETERS``...). See
``docs/architecture/SMART_MICROSCOPY.md`` for the full contract.

This package is deliberately free of Qt: scripts may run in a separate process.
"""

from ._api import (
    API_VERSION,
    STOP,
    AnalysisContext,
    ExecutionMode,
    FrameInfo,
    ParamSpec,
    Response,
    SyncMode,
)

__all__ = [
    "API_VERSION",
    "STOP",
    "AnalysisContext",
    "ExecutionMode",
    "FrameInfo",
    "ParamSpec",
    "Response",
    "SyncMode",
]
