"""Frame sources for the perception plane.

A :class:`FrameSource` yields :data:`FrameBundle` objects -- one synchronised frame per approach
camera. :class:`OpenCvVideoSource` reads real video files or RTSP streams (lazy ``cv2`` import);
:class:`ScriptedFrameSource` replays a fixed list of bundles for deterministic tests and the demo.
"""

from __future__ import annotations

import abc
from collections.abc import Iterator, Sequence
from typing import Any

from sentinel.contracts.enums import Direction
from sentinel.observability.logging import get_logger

_log = get_logger("sentinel.perception.video")

#: One frame per approach camera for a single instant. Frame type is backend-specific
#: (an ``ndarray`` for OpenCV; anything the detector understands for tests).
FrameBundle = dict[Direction, Any]


class FrameSource(abc.ABC):
    """Yields synchronised per-approach frame bundles until exhausted."""

    @abc.abstractmethod
    def frames(self) -> Iterator[FrameBundle]:
        """Iterate frame bundles. Iteration ends when any camera runs out of frames."""

    def close(self) -> None:  # noqa: B027 - intentional concrete no-op default; subclasses override
        """Release capture resources (no-op by default)."""


class ScriptedFrameSource(FrameSource):
    """Replays a predefined list of frame bundles -- deterministic, dependency-free."""

    def __init__(self, bundles: Sequence[FrameBundle]) -> None:
        self._bundles = list(bundles)

    def frames(self) -> Iterator[FrameBundle]:
        yield from self._bundles


class OpenCvVideoSource(FrameSource):
    """Reads four synchronised video streams with OpenCV (requires the ``perception`` extra)."""

    def __init__(self, sources: dict[Direction, str]) -> None:
        self._sources = sources
        self._captures: dict[Direction, Any] = {}

    def frames(self) -> Iterator[FrameBundle]:
        import cv2  # lazy: only needed for real video ingestion

        self._captures = {d: cv2.VideoCapture(path) for d, path in self._sources.items()}
        _log.info("video_opened", cameras=len(self._captures))
        try:
            while True:
                bundle: FrameBundle = {}
                for direction, cap in self._captures.items():
                    ok, frame = cap.read()
                    if not ok:
                        return
                    bundle[direction] = frame
                yield bundle
        finally:
            self.close()

    def close(self) -> None:
        for cap in self._captures.values():
            cap.release()
        self._captures.clear()


__all__ = ["FrameBundle", "FrameSource", "OpenCvVideoSource", "ScriptedFrameSource"]
