"""The Dashboard Agent: a live, read-only projection of the intersection for the UI."""

from __future__ import annotations

from sentinel.dashboard.agent import DashboardAgent
from sentinel.dashboard.readmodel import LiveSnapshot

__all__ = ["DashboardAgent", "LiveSnapshot"]
