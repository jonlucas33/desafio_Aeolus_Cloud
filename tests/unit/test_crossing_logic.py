"""
Testes unitários para CrossingCounter.

Cobre os 5 edge-cases especificados em TASKS.md 2.1 além de comportamentos
auxiliares (votação de classe, best-crop buffer, contagem total).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.domain import Track


# ── Helper ───────────────────────────────────────────────────────────────────

def make_track(
    track_id: int,
    centroid: tuple[float, float],
    class_name: str = "car",
    class_id: int = 2,
    width: float = 100.0,
    height: float = 100.0,
) -> Track:
    """Cria Track com centroid exato derivando bbox_xyxy automaticamente."""
    cx, cy = centroid
    bbox = np.array(
        [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2],
        dtype=np.float32,
    )
    return Track(
        track_id=track_id,
        bbox_xyxy=bbox,
        confidence=0.85,
        class_id=class_id,
        class_name=class_name,
    )


def make_counter(
    line_points: list | None = None,
    direction: str = "any",
    min_displacement_px: int = 5,
    class_vote_window: int = 15,
):
    from src.counting.crossing_logic import CrossingCounter

    if line_points is None:
        line_points = [[0, 540], [1280, 540]]
    return CrossingCounter(
        line_points=line_points,
        direction=direction,
        min_displacement_px=min_displacement_px,
        class_vote_window=class_vote_window,
    )


# ── Edge-case 1: cruzamento correto ─────────────────────────────────────────

def test_vehicle_crossing_top_to_bottom_is_counted() -> None:
    """Veículo cruzando de cima para baixo deve ser contado."""
    counter = make_counter(direction="top_to_bottom")

    counter.update([make_track(1, centroid=(640, 500))])       # acima
    crossed = counter.update([make_track(1, centroid=(640, 580))])  # abaixo

    assert 1 in crossed, "track_id=1 deve aparecer na lista de cruzamentos"
    assert counter.count == 1


# ── Edge-case 2: mesmo track_id não contado duas vezes ──────────────────────

def test_same_track_id_counted_only_once() -> None:
    """Mesmo track_id que cruzar novamente não deve incrementar o contador."""
    counter = make_counter(direction="any")

    counter.update([make_track(1, centroid=(640, 500))])
    counter.update([make_track(1, centroid=(640, 580))])  # 1ª travessia → conta

    # Simula retorno para cima e nova descida (2ª travessia)
    counter.update([make_track(1, centroid=(640, 500))])
    crossed = counter.update([make_track(1, centroid=(640, 580))])

    assert 1 not in crossed, "Segundo cruzamento não deve gerar evento"
    assert counter.count == 1, "Contador deve permanecer em 1"


# ── Edge-case 3: jitter abaixo do limiar não conta ──────────────────────────

def test_vehicle_jitter_below_displacement_threshold_not_counted() -> None:
    """Deslocamento < min_displacement_px ao cruzar a linha não deve ser contado."""
    counter = make_counter(min_displacement_px=5, direction="any")

    # Frame 0: centroid 1 px acima da linha (y=540)
    counter.update([make_track(1, centroid=(640, 539))])

    # Frame 1: centroid 1 px abaixo — deslocamento = sqrt(1²+2²) ≈ 2.24 < 5px
    crossed = counter.update([make_track(1, centroid=(641, 541))])

    assert crossed == [], "Jitter não deve produzir evento de cruzamento"
    assert counter.count == 0


# ── Edge-case 4: linha diagonal detectada corretamente ──────────────────────

def test_diagonal_line_crossing_detected_correctly() -> None:
    """Produto vetorial deve detectar cruzamento em linha não-horizontal."""
    # Linha diagonal: A=(0,0) B=(100,100)
    # _side(0,0,100,100,px,py) = 100*(py-px)
    # Ponto (20,50): positivo  →  Ponto (50,20): negativo  →  cruzamento
    counter = make_counter(
        line_points=[[0, 0], [100, 100]],
        direction="any",
        min_displacement_px=5,
    )

    counter.update([make_track(1, centroid=(20, 50))])   # lado positivo
    crossed = counter.update([make_track(1, centroid=(50, 20))])  # lado negativo

    assert 1 in crossed, "Cruzamento de linha diagonal deve ser detectado"
    assert counter.count == 1


# ── Edge-case 5: direção errada não conta ───────────────────────────────────

def test_wrong_direction_not_counted_with_top_to_bottom_filter() -> None:
    """Veículo de baixo para cima não deve ser contado com direction=top_to_bottom."""
    counter = make_counter(direction="top_to_bottom")

    counter.update([make_track(1, centroid=(640, 580))])   # abaixo da linha
    crossed = counter.update([make_track(1, centroid=(640, 500))])  # subindo

    assert crossed == [], "Movimento bottom-to-top não deve contar com top_to_bottom"
    assert counter.count == 0


# ── Testes adicionais ────────────────────────────────────────────────────────

def test_count_property_reflects_total_crossings() -> None:
    """count deve somar todos os track_ids distintos que cruzaram."""
    counter = make_counter()

    counter.update([make_track(1, centroid=(640, 500)), make_track(2, centroid=(640, 500))])
    counter.update([make_track(1, centroid=(640, 580)), make_track(2, centroid=(640, 580))])

    assert counter.count == 2


def test_get_vehicle_class_returns_majority_vote() -> None:
    """get_vehicle_class deve retornar a classe com mais votos."""
    counter = make_counter()

    for _ in range(3):
        counter.update([make_track(1, centroid=(640, 400), class_name="car")])
    for _ in range(2):
        counter.update([make_track(1, centroid=(640, 410), class_name="motorcycle")])

    assert counter.get_vehicle_class(1) == "car"


def test_get_vehicle_class_returns_unknown_for_missing_id() -> None:
    """get_vehicle_class deve retornar 'unknown' para track_id sem votos."""
    counter = make_counter()
    assert counter.get_vehicle_class(999) == "unknown"


def test_best_crop_buffer_stores_largest_bbox_crop() -> None:
    """_best_crop deve ser substituído quando bbox maior é encontrada."""
    counter = make_counter()
    frame = np.ones((600, 1280, 3), dtype=np.uint8) * 128

    # Frame 0: bbox pequena (50×50)
    counter.update([make_track(1, centroid=(640, 400), width=50, height=50)], frame=frame)
    # Frame 1: bbox maior (200×200) — deve substituir o crop anterior
    big_frame = np.ones((600, 1280, 3), dtype=np.uint8) * 200
    counter.update([make_track(1, centroid=(640, 400), width=200, height=200)], frame=big_frame)

    crop = counter.get_best_crop(1)
    assert crop is not None
    # O crop do frame maior tem pixels=200, não 128
    assert crop[0, 0, 0] == 200, "O crop deve vir do frame com bbox maior"


def test_best_crop_not_replaced_by_smaller_bbox() -> None:
    """Crop existente não deve ser substituído quando nova bbox é menor."""
    counter = make_counter()
    big_frame = np.ones((600, 1280, 3), dtype=np.uint8) * 200
    small_frame = np.ones((600, 1280, 3), dtype=np.uint8) * 100

    # Frame 0: bbox grande
    counter.update([make_track(1, centroid=(640, 400), width=200, height=200)], frame=big_frame)
    # Frame 1: bbox menor — não deve substituir
    counter.update([make_track(1, centroid=(640, 400), width=50, height=50)], frame=small_frame)

    crop = counter.get_best_crop(1)
    assert crop[0, 0, 0] == 200, "O crop do frame maior deve ser preservado"


def test_crossed_ids_returned_with_best_crop_available() -> None:
    """No momento do cruzamento, o best_crop deve estar disponível via get_best_crop."""
    counter = make_counter(direction="any")
    frame = np.ones((600, 1280, 3), dtype=np.uint8) * 77

    counter.update([make_track(1, centroid=(640, 500))], frame=frame)
    crossed = counter.update([make_track(1, centroid=(640, 580))], frame=frame)

    assert 1 in crossed
    crop = counter.get_best_crop(1)
    assert crop is not None, "best_crop deve estar disponível no momento do cruzamento"
