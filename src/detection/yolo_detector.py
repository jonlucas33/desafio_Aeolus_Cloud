"""
Wrapper do modelo YOLOv8 para detecção de veículos.

Responsabilidade única: encapsular a inferência YOLO e converter resultados
para List[Detection] filtrando apenas as classes de veículos COCO relevantes.
"""
from __future__ import annotations

import logging

import numpy as np
from ultralytics import YOLO

from src.config import ModelSettings
from src.domain import Detection

logger = logging.getLogger(__name__)

# Classes COCO relevantes para contagem de veículos em rodovias.
_VEHICLE_CLASS_IDS: dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


class YoloDetector:
    """Encapsula o modelo YOLOv8 e retorna List[Detection] por frame.

    Args:
        settings: Configurações do modelo (pesos, device, thresholds).
    """

    def __init__(self, settings: ModelSettings) -> None:
        self._settings = settings
        self._model = YOLO(settings.weights)
        logger.info(
            "YoloDetector inicializado (device=%s, fp16=%s)",
            settings.device,
            settings.fp16,
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Executa inferência em um frame e retorna apenas detecções de veículos.

        Usa filtragem assimétrica por classe: o YOLO roda com base_conf_threshold
        (baixo) para não perder motos; a máscara de pós-processamento aplica
        motorcycle_threshold para motos (class_id=3) e default_class_threshold
        para todos os demais veículos.

        Args:
            frame: Frame BGR em formato numpy (H, W, 3).

        Returns:
            Lista de Detection filtrada pelas classes car/motorcycle/bus/truck
            com limiares de confiança por classe aplicados.
        """
        results = self._model(
            frame,
            conf=self._settings.base_conf_threshold,
            iou=self._settings.iou_threshold,
            device=self._settings.device,
            verbose=False,
        )

        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy().astype(int)

            # Máscara assimétrica: limiar brando para motos, rigoroso para demais
            moto_mask = (clss == 3) & (confs >= self._settings.motorcycle_threshold)
            other_mask = (clss != 3) & (confs >= self._settings.default_class_threshold)
            keep = moto_mask | other_mask

            for bbox, conf, cls_id in zip(boxes[keep], confs[keep], clss[keep]):
                if cls_id not in _VEHICLE_CLASS_IDS:
                    continue
                detections.append(Detection(
                    bbox_xyxy=bbox.astype(np.float32),
                    confidence=float(conf),
                    class_id=int(cls_id),
                    class_name=_VEHICLE_CLASS_IDS[cls_id],
                ))

        return detections

    def warmup(self, frame_shape: tuple[int, int, int], n_frames: int = 3) -> None:
        """Aquece a GPU com frames sintéticos do shape exato do vídeo de entrada.

        Deve ser chamado antes do loop principal passando o shape real lido
        via VideoCapture. Shape diferente causa realocação de memória GPU no
        primeiro frame real, eliminando o benefício do warmup.

        Args:
            frame_shape: (height, width, channels) — shape exato do vídeo.
            n_frames: Quantas inferências sintéticas executar.
        """
        synthetic = np.zeros(frame_shape, dtype=np.uint8)
        for _ in range(n_frames):
            self._model(synthetic, device=self._settings.device, verbose=False)
        logger.info("GPU warmup concluído (%d frames, shape=%s)", n_frames, frame_shape)
