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
    suv_aspect_ratio_threshold: float = 0.85,
    truck_area_threshold: float = 0.04,
):
    from src.counting.crossing_logic import CrossingCounter

    if line_points is None:
        line_points = [[0, 540], [1280, 540]]
    return CrossingCounter(
        line_points=line_points,
        direction=direction,
        min_displacement_px=min_displacement_px,
        class_vote_window=class_vote_window,
        suv_aspect_ratio_threshold=suv_aspect_ratio_threshold,
        truck_area_threshold=truck_area_threshold,
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
    """get_vehicle_class deve retornar a classe refinada com mais votos.

    class_id=3 (motorcycle) → "motorcycle" sempre.
    class_id=2 (car) com bbox 100x100 (aspect_ratio=1.0 > 0.85) → "suv_pickup".
    3 votos de motorcycle batem 2 votos de suv_pickup.
    """
    counter = make_counter()

    for _ in range(3):
        counter.update([make_track(1, centroid=(640, 400), class_id=3)])
    for _ in range(2):
        counter.update([make_track(1, centroid=(640, 410), class_id=2)])

    assert counter.get_vehicle_class(1) == "motorcycle"


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


# ── Novos testes: AÇÃO 3 — fórmula estrita do produto vetorial ───────────────

def test_crossing_detected_when_centroid_lands_exactly_on_line() -> None:
    """Produto vetorial blindado: centróide cai EXATAMENTE na linha (d2 == 0).

    Caso: d1 < 0 (acima da linha), d2 == 0 (sobre a linha).
    Fórmula antiga: (False) != (False) == False → cruzamento PERDIDO.
    Fórmula nova:   d1 * d2 < 0  OR  (d1==0) != (d2==0)
                    = -0.0 < 0 [False]  OR  False != True [True]  →  DETECTADO.
    """
    # linha em y=540; centróide começa 20 px acima e cai exatamente em y=540
    counter = make_counter(
        line_points=[[0, 540], [1280, 540]],
        direction="any",
        min_displacement_px=5,
    )
    counter.update([make_track(1, centroid=(640, 520))])           # d1 < 0: acima
    crossed = counter.update([make_track(1, centroid=(640, 540))]) # d2 == 0: sobre a linha

    assert 1 in crossed, (
        "d1 < 0, d2 == 0 deve ser detectado como cruzamento — "
        "a fórmula (d1>0)!=(d2>0) perde este caso"
    )
    assert counter.count == 1


# ── Testes AÇÃO 1: heurística de duas camadas (_refine_class) ────────────────

def test_refine_class_wide_car_bbox_returns_sedan_hatch() -> None:
    """Carro COCO (class_id=2) com bbox larga e achatada (w>h) → 'sedan_hatch'.

    Bbox 200x80: aspect_ratio = 80/200 = 0.40 < 0.85 → sedan_hatch.
    """
    counter = make_counter(suv_aspect_ratio_threshold=0.85, truck_area_threshold=0.04)
    bbox = np.array([0.0, 0.0, 200.0, 80.0], dtype=np.float32)
    result = counter._refine_class(class_id=2, bbox_xyxy=bbox, frame_area=1_000_000.0)
    assert result == "sedan_hatch", (
        f"Bbox larga (aspect_ratio=0.40 < 0.85) deve ser 'sedan_hatch', obteve '{result}'"
    )


def test_refine_class_tall_car_bbox_returns_suv_pickup() -> None:
    """Carro COCO (class_id=2) com bbox alta e estreita (h>w) → 'suv_pickup'.

    Bbox 100x150: aspect_ratio = 150/100 = 1.50 > 0.85 → suv_pickup.
    """
    counter = make_counter(suv_aspect_ratio_threshold=0.85, truck_area_threshold=0.04)
    bbox = np.array([0.0, 0.0, 100.0, 150.0], dtype=np.float32)
    result = counter._refine_class(class_id=2, bbox_xyxy=bbox, frame_area=1_000_000.0)
    assert result == "suv_pickup", (
        f"Bbox alta (aspect_ratio=1.50 > 0.85) deve ser 'suv_pickup', obteve '{result}'"
    )


def test_refine_class_small_truck_bbox_returns_suv_pickup() -> None:
    """Caminhão COCO (class_id=7) com area_ratio pequena → 'suv_pickup'.

    Bbox 100x100 (area=10 000), frame_area=1 000 000 → ratio=0.01 < 0.04.
    Heurística: caminhão pequeno na imagem é na verdade SUV/picape.
    """
    counter = make_counter(suv_aspect_ratio_threshold=0.85, truck_area_threshold=0.04)
    bbox = np.array([0.0, 0.0, 100.0, 100.0], dtype=np.float32)
    result = counter._refine_class(class_id=7, bbox_xyxy=bbox, frame_area=1_000_000.0)
    assert result == "suv_pickup", (
        f"Caminhão com area_ratio=0.01 < 0.04 deve ser 'suv_pickup', obteve '{result}'"
    )


def test_refine_class_large_truck_bbox_returns_truck_bus() -> None:
    """Caminhão COCO (class_id=7) com area_ratio grande → 'truck_bus'.

    Bbox 250x200 (area=50 000), frame_area=1 000 000 → ratio=0.05 > 0.04.
    Heurística: objeto grande na imagem é realmente um caminhão/ônibus.
    """
    counter = make_counter(suv_aspect_ratio_threshold=0.85, truck_area_threshold=0.04)
    bbox = np.array([0.0, 0.0, 250.0, 200.0], dtype=np.float32)
    result = counter._refine_class(class_id=7, bbox_xyxy=bbox, frame_area=1_000_000.0)
    assert result == "truck_bus", (
        f"Caminhão com area_ratio=0.05 > 0.04 deve ser 'truck_bus', obteve '{result}'"
    )


def test_jitter_freeze_preserves_reference_and_enables_crossing_detection() -> None:
    """AÇÃO 2 + 3: centróide congelado pelo jitter + fórmula estrita = cruzamento detectado.

    Sequência:
      Frame 0: registra centróide em (640, 520) — 20 px acima da linha.
      Frames 1-2: movimentos de 3 px cada (< min_displacement_px=10) → JITTER.
                  self._previous_centroids[1] permanece congelado em (640, 520).
      Frame 3: salta para (640, 540) — exatamente sobre a linha.
               Deslocamento a partir do centróide CONGELADO = 20 px > 10 → válido.
               d1 = _side(acima) < 0, d2 = _side(sobre_linha) == 0 → DETECTADO.

    Se o centróide NÃO estivesse congelado (bug de "amnésia"), p_prev estaria
    em (640, 526) e o deslocamento seria 14 px — mas d2 == 0 ainda exigiria
    a fórmula correta. Este teste valida AMBAS as correções em conjunto.
    """
    counter = make_counter(
        line_points=[[0, 540], [1280, 540]],
        direction="any",
        min_displacement_px=10,  # limiar elevado para forçar o freeze
    )

    counter.update([make_track(1, centroid=(640, 520))])  # registra p_prev = (640, 520)
    counter.update([make_track(1, centroid=(640, 523))])  # 3 px < 10 → jitter; p_prev CONGELADO
    counter.update([make_track(1, centroid=(640, 526))])  # 6 px < 10 → jitter; p_prev CONGELADO

    # 20 px a partir do centróide congelado em 520 → deslocamento válido
    # d2 == 0 (cai exatamente em y=540) → requer fórmula estrita
    crossed = counter.update([make_track(1, centroid=(640, 540))])

    assert 1 in crossed, (
        "Cruzamento deve ser detectado: centróide congelado em 520 e deslocamento "
        "acumulado de 20 px > 10 px, com centróide atual exatamente sobre a linha"
    )
    assert counter.count == 1
