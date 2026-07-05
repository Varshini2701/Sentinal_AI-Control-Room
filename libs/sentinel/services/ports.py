"""Ports owned by the service layer."""

from __future__ import annotations

import abc

from sentinel.contracts.value_objects import SignalCommand, SignalState


class SignalActuator(abc.ABC):
    """Applies a safe command to the real or simulated signal controller."""

    @abc.abstractmethod
    async def apply(self, command: SignalCommand) -> SignalState:
        """Apply ``command`` and return the authoritative signal state."""


__all__ = ["SignalActuator"]
