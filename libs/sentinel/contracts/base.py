"""Shared base models and time helpers for the contracts package."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict


def utcnow() -> datetime:
    """Return the current time as a timezone-aware UTC ``datetime``.

    Centralised so every timestamp in the system is UTC and tz-aware, and so tests can
    monkeypatch a single function to obtain deterministic clocks.
    """
    return datetime.now(tz=UTC)


class FrozenModel(BaseModel):
    """Base class for immutable value objects.

    Value objects have no identity and are compared by value; freezing them makes them
    hashable and safe to share across agents and cache layers without defensive copying.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        use_enum_values=False,
        validate_assignment=True,
        ser_json_timedelta="float",
    )


class MutableModel(BaseModel):
    """Base class for mutable payloads (e.g. accumulating snapshots before freezing)."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)
