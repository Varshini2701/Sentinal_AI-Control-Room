"""Object detection port and its implementations.

The :class:`ObjectDetector` port is the seam between the (heavy, non-deterministic) neural network
and the rest of the perception plane. :class:`YoloDetector` wraps Ultralytics YOLO (imported
lazily so the package installs and tests run without ``torch``); :class:`ScriptedDetector` returns
pre-programmed detections for deterministic tests and the offline demo.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import Any, ClassVar

from sentinel.contracts.enums import VehicleClass
from sentinel.contracts.value_objects import BoundingBox, Detection
from sentinel.observability.logging import get_logger

_log = get_logger("sentinel.perception.detector")


class ObjectDetector(abc.ABC):
    """Detects objects in a single image frame."""

    @abc.abstractmethod
    def detect(self, frame: Any) -> list[Detection]:
        """Return the detections in ``frame``. Implementations must not raise on empty frames."""


class ScriptedDetector(ObjectDetector):
    """Returns detections from a pre-programmed script -- deterministic, dependency-free.

    Constructed with a sequence of detection lists; each :meth:`detect` call returns the next entry
    (the last entry repeats once exhausted). A single list may be given to return it every call.
    """

    def __init__(self, script: Sequence[Sequence[Detection]] | Sequence[Detection]) -> None:
        if script and isinstance(script[0], Detection):
            self._script: list[list[Detection]] = [list(script)]  # type: ignore[arg-type]
        else:
            self._script = [list(frame) for frame in script]  # type: ignore[arg-type]
        self._index = 0

    def detect(self, frame: Any) -> list[Detection]:
        if not self._script:
            return []
        entry = self._script[min(self._index, len(self._script) - 1)]
        self._index += 1
        return list(entry)


class YoloDetector(ObjectDetector):
    """Ultralytics YOLO detector. Requires the optional ``perception`` extra (``ultralytics``)."""

    #: COCO class id -> Sentinel vehicle class. Emergency vehicles require a fine-tuned model and
    #: are not present in stock COCO weights (documented in the perception README).
    _COCO_TO_CLASS: ClassVar[dict[int, VehicleClass]] = {
        0: VehicleClass.PEDESTRIAN,
        1: VehicleClass.BICYCLE,
        2: VehicleClass.CAR,
        3: VehicleClass.MOTORCYCLE,
        5: VehicleClass.BUS,
        7: VehicleClass.TRUCK,
    }

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda:0",
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.5,
    ) -> None:
        from ultralytics import YOLO  # lazy: only needed to run real inference

        self._model = YOLO(model_path)
        self._device = device
        self._conf = confidence_threshold
        self._iou = iou_threshold
        _log.info("yolo_loaded", model_path=model_path, device=device)

    def detect(self, frame: Any) -> list[Detection]:
        results = self._model.predict(
            frame, conf=self._conf, iou=self._iou, device=self._device, verbose=False
        )
        detections: list[Detection] = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                vehicle_class = self._COCO_TO_CLASS.get(class_id)
                if vehicle_class is None:
                    continue
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                detections.append(
                    Detection(
                        vehicle_class=vehicle_class,
                        confidence=float(box.conf[0]),
                        box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    )
                )
        return detections


__all__ = ["ObjectDetector", "ScriptedDetector", "YoloDetector"]
