"""
Dataclasses canônicas do domínio do pipeline de contagem de veículos.

Contratos de dados trocados entre módulos. Nunca alterar sem atualizar
todos os consumidores no mesmo commit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np


@dataclass
class Detection:
    """Resultado de detecção de um único objeto por um frame do YOLO.

    Attributes:
        bbox_xyxy: Coordenadas [x1, y1, x2, y2] em float32.
        confidence: Score de confiança entre 0 e 1.
        class_id: ID de classe COCO.
        class_name: Nome legível da classe.
    """

    bbox_xyxy: np.ndarray
    confidence: float
    class_id: int
    class_name: str


@dataclass
class Track:
    """Objeto rastreado com identidade persistente entre frames.

    O centroide é calculado automaticamente via __post_init__ a partir
    de bbox_xyxy — não deve ser passado na construção.

    Attributes:
        track_id: ID único atribuído pelo ByteTrack.
        bbox_xyxy: Coordenadas [x1, y1, x2, y2] em float32.
        confidence: Score de confiança da detecção associada.
        class_id: ID de classe COCO.
        class_name: Nome legível da classe.
        centroid: (cx, cy) — ponto central da bbox, calculado automaticamente.
    """

    track_id: int
    bbox_xyxy: np.ndarray
    confidence: float
    class_id: int
    class_name: str
    centroid: tuple[float, float] = field(init=False)

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = self.bbox_xyxy
        self.centroid = (float((x1 + x2) / 2), float((y1 + y2) / 2))


@dataclass
class VehicleEvent:
    """Evento de cruzamento de linha virtual por um veículo rastreado.

    Attributes:
        track_id: ID do track associado ao evento.
        vehicle_class: Classe de negócio do veículo (car, truck, bus, etc.).
        plate_text: Texto da placa lido via OCR, ou None se não detectado.
        plate_confidence: Confiança do OCR, ou None se não detectado.
        frame_number: Número do frame em que ocorreu o cruzamento.
        timestamp: Momento UTC do cruzamento.
    """

    track_id: int
    vehicle_class: str
    plate_text: str | None
    plate_confidence: float | None
    frame_number: int
    timestamp: datetime
