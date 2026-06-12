"""
Wrapper ByteTrack para rastreamento de veículos entre frames.

Responsabilidade única: manter identidade de objetos entre frames consumindo
List[Detection] e produzindo List[Track]. Acoplamento com outros módulos
somente via contratos de src.domain — nunca importa de detection ou counting.
"""
from __future__ import annotations

import logging

import numpy as np
from supervision import ByteTrack
from supervision import Detections as SvDetections

from src.config import TrackingSettings
from src.domain import Detection, Track

logger = logging.getLogger(__name__)


class ByteTrackWrapper:
    """Mantém identidade de objetos rastreados entre frames via ByteTrack.

    Args:
        settings: Configurações de rastreamento (track_buffer, min_box_area).
    """

    def __init__(self, settings: TrackingSettings) -> None:
        self._settings = settings
        self._tracker = ByteTrack(
            lost_track_buffer=settings.track_buffer,
        )
        logger.info("ByteTrackWrapper inicializado (track_buffer=%d)", settings.track_buffer)

    def update(self, detections: list[Detection], frame: np.ndarray) -> list[Track]:
        """Atualiza o rastreador com as detecções do frame atual.

        Args:
            detections: Detecções do frame atual produzidas pelo YoloDetector.
            frame: Frame BGR original (usado internamente pelo ByteTrack).

        Returns:
            Lista de Track com IDs consistentes entre frames.
        """
        if not detections:
            return []

        # Constrói lookup de class_id → class_name a partir das detecções de entrada.
        # O wrapper nunca conhece o mapeamento COCO — recebe dos contratos de domain.
        class_id_to_name: dict[int, str] = {d.class_id: d.class_name for d in detections}

        sv_dets = SvDetections(
            xyxy=np.array([d.bbox_xyxy for d in detections], dtype=np.float32),
            confidence=np.array([d.confidence for d in detections], dtype=np.float32),
            class_id=np.array([d.class_id for d in detections], dtype=int),
        )

        tracked = self._tracker.update_with_detections(sv_dets)

        tracks: list[Track] = []
        for i, bbox in enumerate(tracked.xyxy):
            cls_id = int(tracked.class_id[i])
            tracks.append(Track(
                track_id=int(tracked.tracker_id[i]),
                bbox_xyxy=bbox.astype(np.float32),
                confidence=float(tracked.confidence[i]),
                class_id=cls_id,
                class_name=class_id_to_name.get(cls_id, "unknown"),
            ))

        return tracks

    def reset(self) -> None:
        """Reinicia o estado interno do rastreador."""
        self._tracker.reset()
        logger.debug("ByteTrackWrapper resetado.")
