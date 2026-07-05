"""Sentinel AI shared foundation libraries.

This package is the single source of truth for the domain contracts and cross-cutting
infrastructure used by every Sentinel AI service:

* :mod:`sentinel.contracts` -- value objects, enums and versioned domain events.
* :mod:`sentinel.messaging` -- the event-bus port and its implementations.
* :mod:`sentinel.config` -- typed, environment-driven configuration.
* :mod:`sentinel.observability` -- logging, metrics and tracing.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
