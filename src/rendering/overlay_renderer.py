"""
Renderização do overlay sobre os frames do pipeline.

Responsabilidade única: desenhar linha virtual, bounding boxes com rótulo
de classe, contador total e FPS sobre uma cópia do frame recebido.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from src.config import RenderingSettings
from src.domain import Track

logger = logging.getLogger(__name__)

# Paleta de cores BGR por classe de veículo
_CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "car": (0, 255, 0),          # verde
    "suv_pickup": (255, 0, 0),   # azul
    "truck": (0, 0, 255),        # vermelho
    "bus": (0, 165, 255),        # laranja
    "motorcycle": (0, 255, 255), # amarelo
}
_DEFAULT_COLOR: tuple[int, int, int] = (200, 200, 200)  # cinza para desconhecidos


class OverlayRenderer:
    """Renderiza informações visuais do pipeline sobre cópias dos frames.

    Nunca modifica o frame original — sempre opera sobre frame.copy().

    Args:
        settings: Configurações de renderização (cor e espessura da linha).
        line_points: Dois pontos [[x1,y1],[x2,y2]] da linha virtual.
    """

    def __init__(
        self,
        settings: RenderingSettings,
        line_points: list[list[int]],
    ) -> None:
        self._line_color: tuple[int, int, int] = tuple(settings.line_color)  # type: ignore[assignment]
        self._line_thickness = settings.line_thickness
        self._pt1 = tuple(line_points[0])  # (x1, y1)
        self._pt2 = tuple(line_points[1])  # (x2, y2)

    def draw(
        self,
        frame: np.ndarray,
        tracks: list[Track],
        count: int,
        fps: float,
    ) -> np.ndarray:
        """Desenha overlay sobre uma cópia do frame e retorna o resultado.

        Args:
            frame: Frame BGR original (não modificado).
            tracks: Tracks ativos no frame atual.
            count: Total acumulado de veículos contados.
            fps: Taxa de frames por segundo atual do pipeline.

        Returns:
            Novo array BGR com overlay desenhado.
        """
        canvas = frame.copy()

        self._draw_virtual_line(canvas)
        for track in tracks:
            self._draw_track(canvas, track)
        self._draw_counter(canvas, count)
        self._draw_fps(canvas, fps)

        return canvas

    # ── Métodos privados de desenho ──────────────────────────────────────

    def _draw_virtual_line(self, canvas: np.ndarray) -> None:
        cv2.line(canvas, self._pt1, self._pt2, self._line_color, self._line_thickness)

    def _draw_track(self, canvas: np.ndarray, track: Track) -> None:
        x1, y1, x2, y2 = (int(v) for v in track.bbox_xyxy)
        color = _CLASS_COLORS.get(track.class_name, _DEFAULT_COLOR)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        label = f"{track.class_name} #{track.track_id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = y1 - 4 if y1 - th - 4 >= 0 else y1 + th + 4
        cv2.rectangle(canvas, (x1, label_y - th - 2), (x1 + tw, label_y + 2), color, -1)
        cv2.putText(
            canvas, label, (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
        )

    def _draw_counter(self, canvas: np.ndarray, count: int) -> None:
        text = f"Contagem: {count}"
        cv2.putText(
            canvas, text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA,
        )

    def _draw_fps(self, canvas: np.ndarray, fps: float) -> None:
        text = f"FPS: {fps:.1f}"
        h, w = canvas.shape[:2]
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(
            canvas, text, (w - tw - 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
        )
