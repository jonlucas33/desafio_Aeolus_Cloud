"""
Thread consumidora de OCR usando fast-alpr (ONNX, especializado em placas).

Alternativa ao OCRWorker baseado em EasyOCR. Selecionável via settings.ocr.engine.
Mantém a mesma interface de fila: consome ocr_queue, produz para db_queue.
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import TYPE_CHECKING

from fast_alpr import ALPR

from src.ocr.plate_ocr import _validate_plate

if TYPE_CHECKING:
    from src.config import OCRSettings

logger = logging.getLogger(__name__)


def _resolve_confidence(raw: float | list[float]) -> float:
    """Normaliza confidence do OcrResult para um único float.

    OcrResult.confidence pode ser float (confiança global) ou list[float]
    (confiança por caractere). Neste último caso, retorna a média.

    Args:
        raw: Valor bruto de OcrResult.confidence.

    Returns:
        Confiança como float escalar.
    """
    if isinstance(raw, list):
        return sum(raw) / len(raw) if raw else 0.0
    return float(raw)


class FastAlprWorker(threading.Thread):
    """OCR Worker usando fast-alpr com modelos ONNX especializados em placas.

    Interface compatível com OCRWorker — substituível em main.py sem alteração
    na lógica de fila. Cada item consumido da ocr_queue tem formato
    (track_id, vehicle_crop, event_meta); o resultado é publicado em db_queue
    com plate_text e plate_confidence preenchidos (ou None se inválido).

    Args:
        ocr_queue: Fila de entrada com tuplas (track_id, vehicle_crop, event_meta).
        db_queue: Fila de saída para DbWriter.
        stop_event: Evento de encerramento global compartilhado com main().
        settings: Configurações de OCR do settings.yaml.
    """

    def __init__(
        self,
        ocr_queue: queue.Queue,
        db_queue: queue.Queue,
        stop_event: threading.Event,
        settings: OCRSettings,
    ) -> None:
        super().__init__(daemon=True, name="FastAlprWorker")
        self._ocr_queue = ocr_queue
        self._db_queue = db_queue
        self._stop_event = stop_event
        self._alpr = ALPR(
            detector_model=settings.alpr_detector_model,
            ocr_model=settings.alpr_ocr_model,
        )
        logger.info(
            "FastAlprWorker inicializado (detector=%s, ocr=%s)",
            settings.alpr_detector_model,
            settings.alpr_ocr_model,
        )

    def run(self) -> None:
        """Loop de OCR: consome crops, processa e encaminha para db_queue."""
        while not self._stop_event.is_set() or not self._ocr_queue.empty():
            try:
                track_id, vehicle_crop, event_meta = self._ocr_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            plate_text: str | None = None
            plate_conf: float | None = None

            try:
                results = self._alpr.predict(vehicle_crop)
                if results:
                    best = results[0]
                    if best.ocr is not None:
                        raw_text = best.ocr.text.upper().replace(" ", "")
                        conf = _resolve_confidence(best.ocr.confidence)
                        validated = _validate_plate(raw_text)
                        if validated is not None:
                            plate_text = validated
                            plate_conf = conf
                            logger.info(
                                "Placa detectada: %s (confiança=%.2f, track_id=%d)",
                                plate_text, plate_conf, track_id,
                            )
                        else:
                            logger.debug(
                                "fast-alpr: '%s' (conf=%.2f) não passou na validação regex",
                                raw_text, conf,
                            )
            except Exception:
                logger.warning(
                    "fast-alpr: erro ao processar track_id=%d — fallback para plate_text=None",
                    track_id, exc_info=True,
                )

            event_meta["plate_text"] = plate_text
            event_meta["plate_confidence"] = plate_conf

            try:
                self._db_queue.put_nowait(event_meta)
            except queue.Full:
                logger.warning(
                    "db_queue cheia — evento de track_id=%d descartado", track_id
                )

        logger.info("FastAlprWorker encerrado")

    def stop_and_join(self, timeout: float = 10.0) -> None:
        """Sinaliza parada e aguarda o encerramento da thread.

        Args:
            timeout: Segundos máximos de espera.
        """
        self._stop_event.set()
        self.join(timeout=timeout)
        logger.info("FastAlprWorker encerrado (join concluído)")
