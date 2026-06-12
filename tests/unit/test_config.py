"""
Testes unitários para o loader de configuração Pydantic.
"""
from __future__ import annotations

from pathlib import Path

import pytest

SETTINGS_YAML = Path(__file__).parent.parent.parent / "config" / "settings.yaml"


def test_settings_yaml_exists() -> None:
    assert SETTINGS_YAML.exists(), f"settings.yaml não encontrado em {SETTINGS_YAML}"


def test_load_settings_returns_settings_object() -> None:
    from src.config import Settings, load_settings

    settings = load_settings(SETTINGS_YAML)
    assert isinstance(settings, Settings)


def test_video_settings_fields() -> None:
    from src.config import load_settings

    s = load_settings(SETTINGS_YAML)
    assert isinstance(s.video.source, str)
    assert isinstance(s.video.output, str)
    assert s.video.resize_width > 0


def test_model_settings_fields() -> None:
    from src.config import load_settings

    s = load_settings(SETTINGS_YAML)
    assert isinstance(s.model.weights, str)
    assert 0.0 < s.model.confidence_threshold < 1.0
    assert 0.0 < s.model.iou_threshold < 1.0
    assert s.model.device in ("cuda", "cpu", "mps")
    assert isinstance(s.model.fp16, bool)


def test_tracking_settings_fields() -> None:
    from src.config import load_settings

    s = load_settings(SETTINGS_YAML)
    assert s.tracking.track_buffer > 0
    assert s.tracking.min_box_area >= 0


def test_counting_settings_fields() -> None:
    from src.config import load_settings

    s = load_settings(SETTINGS_YAML)
    assert len(s.counting.line_points) == 2
    assert len(s.counting.line_points[0]) == 2
    assert len(s.counting.line_points[1]) == 2
    assert s.counting.direction in ("any", "top_to_bottom", "bottom_to_top")
    assert s.counting.min_displacement_px > 0
    assert s.counting.class_vote_window > 0


def test_ocr_settings_fields() -> None:
    from src.config import load_settings

    s = load_settings(SETTINGS_YAML)
    assert isinstance(s.ocr.enabled, bool)
    assert 0.0 < s.ocr.min_bbox_area_ratio < 1.0
    assert len(s.ocr.languages) > 0


def test_database_settings_fields() -> None:
    from src.config import load_settings

    s = load_settings(SETTINGS_YAML)
    assert s.database.backend in ("sqlite", "postgresql")
    assert isinstance(s.database.sqlite_path, str)


def test_load_settings_invalid_path_raises() -> None:
    from src.config import load_settings

    with pytest.raises(FileNotFoundError):
        load_settings(Path("/nonexistent/path/settings.yaml"))
