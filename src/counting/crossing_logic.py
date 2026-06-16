"""
Lógica de cruzamento de linha virtual.

Responsabilidade única: determinar quais track_ids cruzaram a linha neste frame
via produto vetorial 2D, mantendo votação de classe e buffer de melhor crop.
"""
from __future__ import annotations

import logging
from collections import Counter

import numpy as np

from src.domain import Track

logger = logging.getLogger(__name__)


# ── Funções puras de geometria ────────────────────────────────────────────────

def _side(
    ax: float, ay: float,
    bx: float, by: float,
    px: float, py: float,
) -> float:
    """Calcula de qual lado da linha virtual o ponto se encontra.

    Retorna valor positivo, negativo ou zero conforme o produto vetorial 2D.

    Nota: a condição de cruzamento usa `d1 * d2 < 0 or (d1 == 0) != (d2 == 0)`
    em vez de `(d1 > 0) != (d2 > 0)` para capturar corretamente o caso em que
    o centróide cai exatamente sobre a linha — situação que a versão naïve perde
    silenciosamente por causa do comportamento de -0.0 em IEEE 754.
    """
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def _crossed_line(
    line_a: tuple[float, float],
    line_b: tuple[float, float],
    p_prev: tuple[float, float],
    p_curr: tuple[float, float],
) -> bool:
    """True se o ponto mudou de lado em relação à linha virtual.

    Fórmula estrita: d1 * d2 < 0 OR exatamente um dos pontos está sobre a linha.
    Isso captura o caso d1 < 0, d2 == 0 (centróide cai exatamente na linha),
    que a fórmula ingênua (d1 > 0) != (d2 > 0) perde: False != False = False.
    Nota: -0.0 < 0 é False em IEEE 754, portanto d1 * d2 < 0 falha quando
    d2 == 0.0 — a cláusula OR cobre esse caso.
    """
    d1 = _side(*line_a, *line_b, *p_prev)
    d2 = _side(*line_a, *line_b, *p_curr)
    return d1 * d2 < 0 or (d1 == 0) != (d2 == 0)


def _is_valid_movement(
    p_prev: tuple[float, float],
    p_curr: tuple[float, float],
    min_displacement_px: float,
) -> bool:
    """True se deslocamento entre frames supera o limiar anti-jitter."""
    dx = p_curr[0] - p_prev[0]
    dy = p_curr[1] - p_prev[1]
    return (dx ** 2 + dy ** 2) ** 0.5 >= min_displacement_px


def _is_correct_direction(
    p_prev: tuple[float, float],
    p_curr: tuple[float, float],
    direction: str,
) -> bool:
    """True se o movimento está na direção configurada."""
    if direction == "any":
        return True
    dy = p_curr[1] - p_prev[1]
    if direction == "top_to_bottom":
        return dy > 0
    if direction == "bottom_to_top":
        return dy < 0
    return True


# ── CrossingCounter ───────────────────────────────────────────────────────────

class CrossingCounter:
    """Detecta e contabiliza cruzamentos de linha virtual por track_id.

    Utiliza produto vetorial 2D para detecção, votação majoritária para
    classificação final e um buffer de melhor crop para dispatch ao OCR.

    Args:
        line_points: Dois pontos [[x1,y1],[x2,y2]] definindo a linha virtual.
        direction: Direção válida — "any", "top_to_bottom" ou "bottom_to_top".
        min_displacement_px: Deslocamento mínimo para ignorar jitter.
        class_vote_window: Janela de votos para votação majoritária de classe.
    """

    def __init__(
        self,
        line_points: list[list[int]],
        direction: str,
        min_displacement_px: int,
        class_vote_window: int,
        suv_aspect_ratio_threshold: float = 0.85,
        truck_area_threshold: float = 0.04,
    ) -> None:
        self._line_a: tuple[float, float] = (float(line_points[0][0]), float(line_points[0][1]))
        self._line_b: tuple[float, float] = (float(line_points[1][0]), float(line_points[1][1]))
        self._direction = direction
        self._min_displacement_px = min_displacement_px
        self._class_vote_window = class_vote_window
        self._suv_aspect_ratio_threshold = suv_aspect_ratio_threshold
        self._truck_area_threshold = truck_area_threshold

        # Idempotência: cada track_id é contado no máximo uma vez
        self._crossed_ids: set[int] = set()
        # Centróide do frame anterior por track_id
        self._previous_centroids: dict[int, tuple[float, float]] = {}
        # Votação de classe: Counter por track_id
        self._class_votes: dict[int, Counter] = {}
        # Buffer de melhor crop: crop com maior área de bbox por track_id
        self._best_crop: dict[int, np.ndarray] = {}
        self._max_bbox_area: dict[int, float] = {}
        # Total acumulado
        self._total_count: int = 0

    # ── Heurística de classificação ───────────────────────────────────────

    def _refine_class(
        self,
        class_id: int,
        bbox_xyxy: np.ndarray,
        frame_area: float,
    ) -> str:
        """Refina a classe COCO em categorias de negócio via duas heurísticas.

        Heurística 1 (carros, class_id=2): aspect_ratio = h/w.
            Se h/w > suv_aspect_ratio_threshold → "suv_pickup" (veículo mais alto).
            Caso contrário → "sedan_hatch".

        Heurística 2 (caminhões/ônibus, class_id=5 ou 7): area_ratio = bbox_area / frame_area.
            Se area_ratio < truck_area_threshold → "suv_pickup" (picape grande).
            Caso contrário → "truck_bus".

        Args:
            class_id: Índice da classe COCO retornado pelo detector.
            bbox_xyxy: Array [x1, y1, x2, y2] da detecção.
            frame_area: Área total do frame em pixels (altura × largura).
                        Quando 0, o filtro de área é ignorado e retorna "truck_bus".

        Returns:
            Categoria de negócio: "sedan_hatch", "suv_pickup", "truck_bus",
            "motorcycle" ou "unknown".
        """
        x1, y1, x2, y2 = bbox_xyxy
        w = max(1.0, float(x2 - x1))
        h = max(1.0, float(y2 - y1))

        if class_id == 3:  # motocicleta
            return "motorcycle"

        if class_id in (5, 7):  # ônibus / caminhão
            if frame_area > 0:
                area_ratio = (w * h) / frame_area
                if area_ratio < self._truck_area_threshold:
                    return "suv_pickup"
            return "truck_bus"

        if class_id == 2:  # carro
            if (h / w) > self._suv_aspect_ratio_threshold:
                return "suv_pickup"
            return "sedan_hatch"

        return "unknown"

    # ── Interface pública ─────────────────────────────────────────────────

    def update(
        self,
        tracks: list[Track],
        frame: np.ndarray | None = None,
    ) -> list[int]:
        """Processa tracks do frame atual e retorna IDs que cruzaram a linha.

        Args:
            tracks: Tracks do frame atual produzidos pelo ByteTrackWrapper.
            frame: Frame BGR atual para extração de crop (opcional).
                   Quando fornecido, mantém o buffer _best_crop atualizado.

        Returns:
            Lista de track_ids que cruzaram a linha virtual neste frame.
        """
        crossed_this_frame: list[int] = []
        frame_area = float(frame.shape[0] * frame.shape[1]) if frame is not None else 0.0

        for track in tracks:
            tid = track.track_id
            centroid = track.centroid
            bbox = track.bbox_xyxy

            # 1. Refinar classe e registrar voto
            if tid not in self._class_votes:
                self._class_votes[tid] = Counter()
            refined = self._refine_class(track.class_id, bbox, frame_area)
            self._class_votes[tid][refined] += 1

            # 2. Atualizar best-crop buffer se frame disponível
            if frame is not None:
                bbox_area = float(
                    (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                )
                if bbox_area > self._max_bbox_area.get(tid, -1.0):
                    x1 = max(0, int(bbox[0]))
                    y1 = max(0, int(bbox[1]))
                    x2 = min(frame.shape[1], int(bbox[2]))
                    y2 = min(frame.shape[0], int(bbox[3]))
                    self._best_crop[tid] = frame[y1:y2, x1:x2].copy()
                    self._max_bbox_area[tid] = bbox_area

            # 3. Primeiro frame para este track: apenas registrar centróide
            if tid not in self._previous_centroids:
                self._previous_centroids[tid] = centroid
                continue

            # 4. Já contado: apenas atualizar centróide
            if tid in self._crossed_ids:
                self._previous_centroids[tid] = centroid
                continue

            p_prev = self._previous_centroids[tid]
            p_curr = centroid

            # 5. Filtro de jitter — INVARIANTE: centróide NÃO é atualizado.
            #    self._previous_centroids[tid] permanece congelado em p_prev até que
            #    o veículo acumule deslocamento ≥ min_displacement_px. Isso é essencial
            #    para detectar travessias lentas: cada frame de jitter NÃO avança a
            #    referência, então uma travessia futura ainda será medida a partir do
            #    último ponto válido — não do último ponto de jitter.
            if not _is_valid_movement(p_prev, p_curr, self._min_displacement_px):
                continue  # centróide congelado; update em linha 190 NÃO é alcançado

            # 6. Filtro de direção
            if not _is_correct_direction(p_prev, p_curr, self._direction):
                self._previous_centroids[tid] = centroid
                continue

            # 7. Detecção de cruzamento via produto vetorial 2D
            if _crossed_line(self._line_a, self._line_b, p_prev, p_curr):
                crossed_this_frame.append(tid)
                self._crossed_ids.add(tid)
                self._total_count += 1
                logger.info("Cruzamento detectado: track_id=%d classe=%s", tid, track.class_name)

            self._previous_centroids[tid] = centroid

        return crossed_this_frame

    def get_vehicle_class(self, track_id: int) -> str:
        """Retorna a classe com mais votos para o track_id informado.

        Args:
            track_id: ID do track rastreado.

        Returns:
            Nome da classe majoritária, ou "unknown" se sem histórico.
        """
        votes = self._class_votes.get(track_id)
        if not votes:
            return "unknown"
        return votes.most_common(1)[0][0]

    def get_class_counts(self) -> dict[str, int]:
        """Retorna contagem por classe de veículo para todos os track_ids que cruzaram.

        Usa a votação majoritária de cada track_id cruzado para determinar sua classe.

        Returns:
            Dicionário {class_name: count} com os totais acumulados na sessão.
        """
        counts: dict[str, int] = {}
        for tid in self._crossed_ids:
            cls = self.get_vehicle_class(tid)
            counts[cls] = counts.get(cls, 0) + 1
        return counts

    def get_best_crop(self, track_id: int) -> np.ndarray | None:
        """Retorna o crop de maior área visto para o track_id, ou None.

        Usado pelo main.py para despachar o melhor crop ao OCR no cruzamento.

        Args:
            track_id: ID do track rastreado.

        Returns:
            Array BGR com o crop, ou None se nenhum frame foi fornecido ainda.
        """
        return self._best_crop.get(track_id)

    @property
    def count(self) -> int:
        """Total de veículos distintos contados na sessão."""
        return self._total_count
