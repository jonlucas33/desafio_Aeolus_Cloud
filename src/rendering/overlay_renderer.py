"""
Renderização do overlay sobre os frames do pipeline.

Responsabilidade única: desenhar linha virtual, bounding boxes com rótulo
de classe refinada, contador total, FPS e legenda de cores sobre uma cópia
do frame recebido.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np
import supervision as sv

from src.config import RenderingSettings
from src.domain import Track

logger = logging.getLogger(__name__)


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    """Converte cor hexadecimal para BGR do OpenCV via sv.Color.

    Args:
        hex_color: String no formato "#RRGGBB".

    Returns:
        Tupla BGR para uso direto em funções cv2.
    """
    c = sv.Color.from_hex(hex_color)
    return (c.b, c.g, c.r)


# Paleta de cores BGR por classe refinada do CrossingCounter.
# Chaves correspondem às strings retornadas por CrossingCounter.get_vehicle_class().
_CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "sedan_hatch": _hex_to_bgr("#00C850"),  # Verde
    "suv_pickup":  _hex_to_bgr("#FF8C00"),  # Laranja Forte
    "truck_bus":   _hex_to_bgr("#DC0000"),  # Vermelho Escuro
    "motorcycle":  _hex_to_bgr("#00D7FF"),  # Azul Ciano
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
        class_overrides: dict[int, str] | None = None,
    ) -> np.ndarray:
        """Desenha overlay sobre uma cópia do frame e retorna o resultado.

        Args:
            frame: Frame BGR original (não modificado).
            tracks: Tracks ativos no frame atual.
            count: Total acumulado de veículos contados.
            fps: Taxa de frames por segundo atual do pipeline.
            class_overrides: Mapeamento opcional {track_id → refined_class_name}
                fornecido pelo CrossingCounter via main.py. Quando presente,
                substitui track.class_name na escolha de cor e rótulo da bbox.
                Garante que as cores reflitam a classificação refinada
                (sedan_hatch, suv_pickup, truck_bus, motorcycle) em vez das
                classes COCO brutas do detector.

        Returns:
            Novo array BGR com overlay desenhado.
        """
        canvas = frame.copy()
        overrides = class_overrides or {}

        self._draw_virtual_line(canvas)
        for track in tracks:
            cls_name = overrides.get(track.track_id, track.class_name)
            self._draw_track(canvas, track, cls_name)
        self._draw_counter(canvas, count)
        self._draw_fps(canvas, fps)
        self._draw_legend(canvas)

        return canvas

    # ── Métodos privados de desenho ──────────────────────────────────────

    def _draw_virtual_line(self, canvas: np.ndarray) -> None:
        cv2.line(canvas, self._pt1, self._pt2, self._line_color, self._line_thickness)

    def _draw_track(
        self,
        canvas: np.ndarray,
        track: Track,
        class_name: str,
    ) -> None:
        """Desenha bbox, background de rótulo e texto da classe refinada.

        Args:
            canvas: Frame em que o overlay é desenhado (in-place).
            track: Track com coordenadas da bbox.
            class_name: Nome de classe refinado (ex.: "sedan_hatch") para
                determinar a cor e o rótulo exibido no vídeo.
        """
        x1, y1, x2, y2 = (int(v) for v in track.bbox_xyxy)
        color = _CLASS_COLORS.get(class_name, _DEFAULT_COLOR)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        label = f"{class_name} #{track.track_id}"
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

    def _draw_legend(self, canvas: np.ndarray) -> None:
        """Desenha legenda de cores por classe refinada no canto inferior esquerdo."""
        swatch_size = 14
        row_height = 20
        x0 = 10
        h = canvas.shape[0]
        y0 = h - len(_CLASS_COLORS) * row_height - 8

        for i, (cls_name, color) in enumerate(_CLASS_COLORS.items()):
            y = y0 + i * row_height
            cv2.rectangle(canvas, (x0, y), (x0 + swatch_size, y + swatch_size), color, -1)
            cv2.putText(
                canvas, cls_name,
                (x0 + swatch_size + 6, y + swatch_size - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA,
            )
