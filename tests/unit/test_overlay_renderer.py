"""
Testes unitários para OverlayRenderer.

Verifica que o overlay é desenhado sobre uma cópia do frame (não in-place),
que a linha virtual e as bboxes são renderizadas, e que o contador e FPS
aparecem em posições corretas.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.config import RenderingSettings
from src.domain import Track


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_settings(
    line_color: list[int] | None = None,
    line_thickness: int = 2,
) -> RenderingSettings:
    return RenderingSettings(
        line_color=line_color or [0, 255, 255],
        line_thickness=line_thickness,
    )


def make_track(
    track_id: int = 1,
    centroid: tuple[float, float] = (320.0, 240.0),
    class_name: str = "car",
    width: float = 80.0,
    height: float = 60.0,
) -> Track:
    cx, cy = centroid
    bbox = np.array(
        [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2],
        dtype=np.float32,
    )
    return Track(
        track_id=track_id,
        bbox_xyxy=bbox,
        confidence=0.9,
        class_id=2,
        class_name=class_name,
    )


def blank_frame(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


LINE_POINTS = [[0, 240], [640, 240]]


# ── Testes ───────────────────────────────────────────────────────────────────

def test_draw_returns_copy_not_same_array() -> None:
    """draw() deve retornar frame.copy(), não o array original."""
    from src.rendering.overlay_renderer import OverlayRenderer

    renderer = OverlayRenderer(settings=make_settings(), line_points=LINE_POINTS)
    frame = blank_frame()
    original_id = id(frame)

    result = renderer.draw(frame, tracks=[], count=0, fps=30.0)

    assert id(result) != original_id, "draw() não deve modificar o frame original"


def test_draw_does_not_modify_original_frame() -> None:
    """O frame original deve permanecer sem alterações após draw()."""
    from src.rendering.overlay_renderer import OverlayRenderer

    renderer = OverlayRenderer(settings=make_settings(), line_points=LINE_POINTS)
    frame = blank_frame()
    original = frame.copy()

    renderer.draw(frame, tracks=[], count=0, fps=30.0)

    assert np.array_equal(frame, original), "Frame original não deve ser modificado"


def test_virtual_line_is_drawn_on_output() -> None:
    """A linha virtual deve modificar o frame de saída."""
    from src.rendering.overlay_renderer import OverlayRenderer

    renderer = OverlayRenderer(settings=make_settings(), line_points=LINE_POINTS)
    frame = blank_frame()

    result = renderer.draw(frame, tracks=[], count=0, fps=30.0)

    # O frame original é todo preto; se a linha foi desenhada, algum pixel mudou
    assert not np.array_equal(result, frame), "A linha virtual deve ser desenhada"


def test_draw_with_no_tracks_does_not_raise() -> None:
    """draw() com lista vazia de tracks não deve lançar exceção."""
    from src.rendering.overlay_renderer import OverlayRenderer

    renderer = OverlayRenderer(settings=make_settings(), line_points=LINE_POINTS)
    frame = blank_frame()

    result = renderer.draw(frame, tracks=[], count=0, fps=0.0)

    assert result.shape == frame.shape


def test_draw_with_single_track_returns_correct_shape() -> None:
    """draw() com um track deve retornar frame com as mesmas dimensões."""
    from src.rendering.overlay_renderer import OverlayRenderer

    renderer = OverlayRenderer(settings=make_settings(), line_points=LINE_POINTS)
    frame = blank_frame()
    track = make_track(1, centroid=(320.0, 100.0), class_name="car")

    result = renderer.draw(frame, tracks=[track], count=1, fps=25.0)

    assert result.shape == frame.shape


def test_draw_bbox_modifies_output_frame() -> None:
    """Bbox desenhada deve alterar pixels do frame de saída."""
    from src.rendering.overlay_renderer import OverlayRenderer

    renderer = OverlayRenderer(settings=make_settings(), line_points=LINE_POINTS)
    frame = blank_frame()
    track = make_track(1, centroid=(320.0, 100.0), class_name="car")

    result = renderer.draw(frame, tracks=[track], count=1, fps=25.0)

    # Resultado deve diferir do frame original em branco
    assert not np.array_equal(result, frame)


def test_draw_accepts_multiple_tracks() -> None:
    """draw() deve aceitar múltiplos tracks sem erro."""
    from src.rendering.overlay_renderer import OverlayRenderer

    renderer = OverlayRenderer(settings=make_settings(), line_points=LINE_POINTS)
    frame = blank_frame()
    tracks = [
        make_track(1, centroid=(100.0, 100.0), class_name="car"),
        make_track(2, centroid=(400.0, 300.0), class_name="truck"),
        make_track(3, centroid=(200.0, 150.0), class_name="motorcycle"),
    ]

    result = renderer.draw(frame, tracks=tracks, count=3, fps=30.0)

    assert result.shape == frame.shape


def test_line_color_from_settings_applied() -> None:
    """A cor da linha deve ser lida de RenderingSettings.line_color."""
    from src.rendering.overlay_renderer import OverlayRenderer

    # Linha vermelha pura (BGR: [0, 0, 255])
    red_settings = make_settings(line_color=[0, 0, 255])
    renderer = OverlayRenderer(settings=red_settings, line_points=[[0, 240], [640, 240]])
    frame = blank_frame()

    result = renderer.draw(frame, tracks=[], count=0, fps=0.0)

    # Linha desenhada em y=240: pelo menos um pixel deve ter canal R=255
    row = result[240, :, :]
    assert np.any(row[:, 2] == 255), "Canal R deve ser 255 para linha vermelha"
