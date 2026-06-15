"""
Detector de placas veiculares — segundo estágio do pipeline OCR.

Executa um modelo YOLOv8 treinado para detecção de placas sobre crops
veiculares, retornando o recorte da placa de maior confiança com margem
de segurança aplicada.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

_MIN_HEIGHT_FOR_INFERENCE: int = 64   # px; abaixo disso o crop é upscalado
_MARGIN_PX: int = 4                   # margem de segurança em torno da bbox


class PlateDetector:
    """Detecta regiões de placa em crops veiculares via YOLOv8.

    Args:
        weights_path: Caminho para o arquivo .pt do modelo de detecção de placas.
        conf_threshold: Confiança mínima para aceitar uma detecção (padrão: 0.5).
    """

    def __init__(self, weights_path: str, conf_threshold: float = 0.5) -> None:
        self._model = YOLO(weights_path)
        self._conf_threshold = conf_threshold
        logger.info(
            "PlateDetector inicializado: %s (conf=%.2f)", weights_path, conf_threshold
        )

    def detect(self, vehicle_crop: np.ndarray) -> np.ndarray | None:
        """Detecta a placa com maior confiança no crop veicular.

        Se a altura do crop for inferior a 64 px, faz upscale CUBIC antes da
        inferência e mapeia as coordenadas detectadas de volta ao espaço original
        antes de recortar — preservando qualidade e evitando artefatos.

        Args:
            vehicle_crop: Frame BGR recortado ao redor do veículo.

        Returns:
            Crop BGR da placa (com margem de 4 px clampada), ou None se não
            encontrada ou se a melhor detecção não atingir o threshold.
        """
        h_orig, w_orig = vehicle_crop.shape[:2]

        # Upscale para melhorar detecção em crops pequenos
        scale = 1.0
        inference_frame = vehicle_crop
        if h_orig < _MIN_HEIGHT_FOR_INFERENCE:
            scale = _MIN_HEIGHT_FOR_INFERENCE / h_orig
            new_w = max(1, int(w_orig * scale))
            inference_frame = cv2.resize(
                vehicle_crop,
                (new_w, _MIN_HEIGHT_FOR_INFERENCE),
                interpolation=cv2.INTER_CUBIC,
            )

        results = self._model(inference_frame, conf=self._conf_threshold, verbose=False)

        best_box: np.ndarray | None = None
        best_conf: float = -1.0

        for result in results:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            for box, conf in zip(boxes_xyxy, confs):
                if float(conf) >= self._conf_threshold and float(conf) > best_conf:
                    best_conf = float(conf)
                    best_box = box

        if best_box is None:
            return None

        # Mapear coordenadas de volta ao espaço original (se houve upscale)
        x1 = int(best_box[0] / scale)
        y1 = int(best_box[1] / scale)
        x2 = int(best_box[2] / scale)
        y2 = int(best_box[3] / scale)

        # Margem de segurança de 4 px, clampada aos limites do crop original
        x1 = max(0, x1 - _MARGIN_PX)
        y1 = max(0, y1 - _MARGIN_PX)
        x2 = min(w_orig, x2 + _MARGIN_PX)
        y2 = min(h_orig, y2 + _MARGIN_PX)

        if x2 <= x1 or y2 <= y1:
            return None

        return vehicle_crop[y1:y2, x1:x2]
