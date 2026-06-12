"""
Testes unitários para as dataclasses canônicas do domínio.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest


def test_detection_has_required_fields() -> None:
    from src.domain import Detection

    bbox = np.array([10.0, 20.0, 110.0, 120.0], dtype=np.float32)
    d = Detection(bbox_xyxy=bbox, confidence=0.9, class_id=2, class_name="car")

    assert np.array_equal(d.bbox_xyxy, bbox)
    assert d.confidence == 0.9
    assert d.class_id == 2
    assert d.class_name == "car"


def test_track_centroid_is_midpoint_of_bbox() -> None:
    from src.domain import Track

    # bbox: x1=0, y1=0, x2=100, y2=80  → centroid=(50.0, 40.0)
    bbox = np.array([0.0, 0.0, 100.0, 80.0], dtype=np.float32)
    t = Track(
        track_id=1,
        bbox_xyxy=bbox,
        confidence=0.85,
        class_id=2,
        class_name="car",
    )

    assert t.centroid == (50.0, 40.0)


def test_track_centroid_is_tuple_of_floats() -> None:
    from src.domain import Track

    bbox = np.array([10.0, 20.0, 50.0, 60.0], dtype=np.float32)
    t = Track(track_id=7, bbox_xyxy=bbox, confidence=0.7, class_id=3, class_name="motorcycle")

    assert isinstance(t.centroid, tuple)
    assert len(t.centroid) == 2
    assert isinstance(t.centroid[0], float)
    assert isinstance(t.centroid[1], float)


def test_track_centroid_calculated_automatically_via_post_init() -> None:
    from src.domain import Track

    # centroid deve ser calculado mesmo sem passá-lo explicitamente
    bbox = np.array([30.0, 40.0, 70.0, 100.0], dtype=np.float32)
    t = Track(track_id=3, bbox_xyxy=bbox, confidence=0.6, class_id=5, class_name="bus")

    expected_cx = (30.0 + 70.0) / 2   # 50.0
    expected_cy = (40.0 + 100.0) / 2  # 70.0
    assert t.centroid == (expected_cx, expected_cy)


def test_vehicle_event_has_required_fields() -> None:
    from src.domain import VehicleEvent

    ts = datetime(2026, 6, 12, 10, 0, 0)
    ev = VehicleEvent(
        track_id=5,
        vehicle_class="truck",
        plate_text="ABC1D23",
        plate_confidence=0.95,
        frame_number=300,
        timestamp=ts,
    )

    assert ev.track_id == 5
    assert ev.vehicle_class == "truck"
    assert ev.plate_text == "ABC1D23"
    assert ev.plate_confidence == 0.95
    assert ev.frame_number == 300
    assert ev.timestamp == ts


def test_vehicle_event_plate_text_can_be_none() -> None:
    from src.domain import VehicleEvent

    ev = VehicleEvent(
        track_id=2,
        vehicle_class="car",
        plate_text=None,
        plate_confidence=None,
        frame_number=10,
        timestamp=datetime.now(),
    )

    assert ev.plate_text is None
    assert ev.plate_confidence is None
