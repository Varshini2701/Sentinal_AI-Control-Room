"""The fast Perception plane: video -> detection -> tracking -> movement -> density -> state.

Pure state-estimation components (geometry, tracking, movement, density, pipeline) have no ML or
video dependency and are fully tested. The detector and video source wrap Ultralytics YOLO and
OpenCV behind ports, imported lazily; the worker ties them together and publishes ``state.updated``.
"""

from __future__ import annotations

from sentinel.perception.density import DensityEstimator
from sentinel.perception.detector import ObjectDetector, ScriptedDetector, YoloDetector
from sentinel.perception.geometry import (
    IntersectionCalibration,
    LaneCalibration,
    default_calibration,
    point_in_polygon,
)
from sentinel.perception.movement import MovementAnalyzer, MovementInfo
from sentinel.perception.pipeline import PerceptionPipeline
from sentinel.perception.tracking import IouTracker, MultiObjectTracker, iou
from sentinel.perception.video import (
    FrameBundle,
    FrameSource,
    OpenCvVideoSource,
    ScriptedFrameSource,
)
from sentinel.perception.worker import PerceptionWorker

__all__ = [
    "DensityEstimator",
    "FrameBundle",
    "FrameSource",
    "IntersectionCalibration",
    "IouTracker",
    "LaneCalibration",
    "MovementAnalyzer",
    "MovementInfo",
    "MultiObjectTracker",
    "ObjectDetector",
    "OpenCvVideoSource",
    "PerceptionPipeline",
    "PerceptionWorker",
    "ScriptedDetector",
    "ScriptedFrameSource",
    "YoloDetector",
    "default_calibration",
    "iou",
    "point_in_polygon",
]
