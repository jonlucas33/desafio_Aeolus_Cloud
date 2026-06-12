"""
Testes unitários para o wrapper ByteTrack.

supervision.ByteTrack é mockado — sem dependência de pesos ou vídeo real.
O acoplamento com YoloDetector é somente via contratos de domain.py.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.config import TrackingSettings
from src.domain import Detection


def _make_settings() -> TrackingSettings:
    return TrackingSettings(track_buffer=30, min_box_area=100)


def _make_detection(
    bbox: list[float] = None,
    class_id: int = 2,
    class_name: str = "car",
    confidence: float = 0.85,
) -> Detection:
    if bbox is None:
        bbox = [10.0, 20.0, 100.0, 200.0]
    return Detection(
        bbox_xyxy=np.array(bbox, dtype=np.float32),
        confidence=confidence,
        class_id=class_id,
        class_name=class_name,
    )


def _make_tracked_sv(
    xyxy: list[list[float]],
    tracker_ids: list[int],
    class_ids: list[int],
    confidences: list[float],
) -> MagicMock:
    """Mock de sv.Detections retornado pelo tracker após update."""
    tracked = MagicMock()
    tracked.xyxy = np.array(xyxy, dtype=np.float32)
    tracked.tracker_id = np.array(tracker_ids, dtype=int)
    tracked.class_id = np.array(class_ids, dtype=int)
    tracked.confidence = np.array(confidences, dtype=np.float32)
    return tracked


# ---------------------------------------------------------------------------

def test_update_returns_list_of_tracks(mocker) -> None:
    """update() deve retornar List[Track] com os campos corretos."""
    mock_tracker = MagicMock()
    mocker.patch("src.tracking.bytetrack_wrapper.ByteTrack", return_value=mock_tracker)
    mock_tracker.update_with_detections.return_value = _make_tracked_sv(
        xyxy=[[10.0, 20.0, 100.0, 200.0]],
        tracker_ids=[1],
        class_ids=[2],
        confidences=[0.85],
    )

    from src.tracking.bytetrack_wrapper import ByteTrackWrapper
    from src.domain import Track

    wrapper = ByteTrackWrapper(_make_settings())
    detections = [_make_detection()]
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    tracks = wrapper.update(detections, frame)

    assert len(tracks) == 1
    assert isinstance(tracks[0], Track)
    assert tracks[0].track_id == 1
    assert tracks[0].class_id == 2
    assert tracks[0].class_name == "car"
    assert tracks[0].confidence == pytest.approx(0.85)


def test_update_centroid_is_calculated_automatically(mocker) -> None:
    """centroid deve ser calculado via Track.__post_init__, não no wrapper."""
    mock_tracker = MagicMock()
    mocker.patch("src.tracking.bytetrack_wrapper.ByteTrack", return_value=mock_tracker)
    # bbox: x1=0, y1=0, x2=100, y2=80 → centroid=(50.0, 40.0)
    mock_tracker.update_with_detections.return_value = _make_tracked_sv(
        xyxy=[[0.0, 0.0, 100.0, 80.0]],
        tracker_ids=[5],
        class_ids=[7],
        confidences=[0.9],
    )

    from src.tracking.bytetrack_wrapper import ByteTrackWrapper

    wrapper = ByteTrackWrapper(_make_settings())
    tracks = wrapper.update(
        [_make_detection(bbox=[0.0, 0.0, 100.0, 80.0], class_id=7, class_name="truck")],
        np.zeros((480, 640, 3), dtype=np.uint8),
    )

    assert tracks[0].centroid == (50.0, 40.0)


def test_update_with_empty_detections_returns_empty_list(mocker) -> None:
    """update() com lista vazia de detecções deve retornar lista vazia."""
    mock_tracker = MagicMock()
    mocker.patch("src.tracking.bytetrack_wrapper.ByteTrack", return_value=mock_tracker)

    from src.tracking.bytetrack_wrapper import ByteTrackWrapper

    wrapper = ByteTrackWrapper(_make_settings())
    tracks = wrapper.update([], np.zeros((480, 640, 3), dtype=np.uint8))

    assert tracks == []
    mock_tracker.update_with_detections.assert_not_called()


def test_update_preserves_class_name_from_input_detections(mocker) -> None:
    """class_name no Track deve ser propagado das Detection de entrada."""
    mock_tracker = MagicMock()
    mocker.patch("src.tracking.bytetrack_wrapper.ByteTrack", return_value=mock_tracker)
    mock_tracker.update_with_detections.return_value = _make_tracked_sv(
        xyxy=[[0.0, 0.0, 50.0, 50.0], [60.0, 60.0, 120.0, 120.0]],
        tracker_ids=[10, 11],
        class_ids=[5, 7],
        confidences=[0.7, 0.8],
    )

    from src.tracking.bytetrack_wrapper import ByteTrackWrapper

    wrapper = ByteTrackWrapper(_make_settings())
    detections = [
        _make_detection(class_id=5, class_name="bus"),
        _make_detection(class_id=7, class_name="truck"),
    ]
    tracks = wrapper.update(detections, np.zeros((480, 640, 3), dtype=np.uint8))

    names_by_id = {t.class_id: t.class_name for t in tracks}
    assert names_by_id[5] == "bus"
    assert names_by_id[7] == "truck"


def test_reset_calls_underlying_tracker_reset(mocker) -> None:
    """reset() deve delegar ao tracker interno para limpar estado."""
    mock_tracker = MagicMock()
    mocker.patch("src.tracking.bytetrack_wrapper.ByteTrack", return_value=mock_tracker)

    from src.tracking.bytetrack_wrapper import ByteTrackWrapper

    wrapper = ByteTrackWrapper(_make_settings())
    wrapper.reset()

    mock_tracker.reset.assert_called_once()
