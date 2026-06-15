"""
OCR de placas veiculares com EasyOCR e thread consumidora de ocr_queue.

Responsabilidade única: pré-processar crops de veículos, executar OCR de dois
estágios (PlateDetector → EasyOCR), validar padrões de placa brasileira e
encaminhar resultado para db_queue.
"""
from __future__ import annotations

import logging
import queue
import re
import threading
from typing import TYPE_CHECKING

import cv2
import easyocr
import numpy as np

if TYPE_CHECKING:
    from src.ocr.plate_detector import PlateDetector

logger = logging.getLogger(__name__)

# Padrões de placa brasileira
_MERCOSUL = re.compile(r"^[A-Z]{3}[0-9][A-Z][0-9]{2}$")
_OLD_FORMAT = re.compile(r"^[A-Z]{3}[0-9]{4}$")
_MIN_CONFIDENCE: float = 0.3
_MIN_CROP_HEIGHT_PX: int = 100


def _validate_plate(text: str) -> str | None:
    """Normaliza e valida texto contra os padrões de placa brasileira.

    Args:
        text: Texto bruto retornado pelo OCR.

    Returns:
        Texto da placa em maiúsculas sem hifens/espaços, ou None se inválido.
    """
    cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
    if _MERCOSUL.match(cleaned) or _OLD_FORMAT.match(cleaned):
        return cleaned
    return None


def _preprocess_crop(crop: np.ndarray) -> np.ndarray:
    """Redimensiona, converte para cinza e aplica CLAHE para melhorar contraste.

    Args:
        crop: Frame BGR recortado ao redor do veículo.

    Returns:
        Imagem em escala de cinza pré-processada, pronta para o EasyOCR.
    """
    h, w = crop.shape[:2]
    if h < _MIN_CROP_HEIGHT_PX:
        scale = _MIN_CROP_HEIGHT_PX / h
        crop = cv2.resize(
            crop,
            (max(1, int(w * scale)), _MIN_CROP_HEIGHT_PX),
            interpolation=cv2.INTER_CUBIC,
        )

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


class PlateOCR:
    """Leitor de placas baseado em EasyOCR com pré-processamento de imagem.

    Args:
        languages: Lista de idiomas para o EasyOCR (padrão: ['en']).
        gpu: Se True, usa GPU para inferência (padrão: False).
        confidence_threshold: Confiança mínima para aceitar um resultado de OCR.
    """

    def __init__(
        self,
        languages: list[str] | None = None,
        gpu: bool = False,
        confidence_threshold: float = _MIN_CONFIDENCE,
    ) -> None:
        self._languages = languages or ["en"]
        self._min_confidence = confidence_threshold
        self._reader = easyocr.Reader(self._languages, gpu=gpu, verbose=False)
        logger.info(
            "PlateOCR inicializado (idiomas=%s, gpu=%s, conf_threshold=%.2f)",
            self._languages, gpu, self._min_confidence,
        )

    def read(self, crop: np.ndarray) -> tuple[str | None, float | None]:
        """Processa um crop e retorna a placa detectada (se válida).

        Pipeline: pré-processamento → EasyOCR → resultado de maior confiança
        → validação regex → retorno.

        Args:
            crop: Imagem BGR do recorte do veículo.

        Returns:
            Tupla (plate_text, confidence) ou (None, None) se inválido.
        """
        preprocessed = _preprocess_crop(crop)

        try:
            results = self._reader.readtext(preprocessed)
        except Exception:
            logger.warning("EasyOCR: falha ao processar crop", exc_info=True)
            return None, None

        if not results:
            return None, None

        # Pegar o candidato de maior confiança
        _, text, confidence = max(results, key=lambda r: r[2])

        if confidence < self._min_confidence:
            return None, None

        plate = _validate_plate(text)
        if plate is None:
            logger.debug("OCR: '%s' (conf=%.2f) não passou na validação regex", text, confidence)
            return None, None

        logger.info("Placa detectada: %s (confiança=%.2f)", plate, confidence)
        return plate, float(confidence)


class OCRWorker:
    """Thread consumidora da ocr_queue que despacha resultados para db_queue.

    Implementa OCR de dois estágios:
      1. PlateDetector localiza a placa no crop do veículo (quando disponível).
      2. EasyOCR lê o texto da placa (usando o crop do Estágio 1 ou fallback).

    Cada item da ocr_queue tem formato: (track_id, crop, event_meta).
    event_meta é um dict com os campos necessários para VehicleEventModel,
    exceto plate_text e plate_confidence (adicionados pelo worker).

    Args:
        ocr_queue: Fila de entrada (track_id, crop, event_meta).
        db_queue: Fila de saída para DbWriter.
        plate_ocr: Instância de PlateOCR para inferência EasyOCR.
        plate_detector: Detector de placa opcional para o estágio 1. Quando
            None, o OCR opera diretamente sobre o crop do veículo (modo legado).
    """

    def __init__(
        self,
        ocr_queue: queue.Queue,
        db_queue: queue.Queue,
        plate_ocr: PlateOCR,
        plate_detector: PlateDetector | None = None,
    ) -> None:
        self._ocr_queue = ocr_queue
        self._db_queue = db_queue
        self._plate_ocr = plate_ocr
        self._plate_detector = plate_detector
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Inicia a thread daemon de OCR."""
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="OCRWorker"
        )
        self._thread.start()
        logger.info("OCRWorker iniciado")

    def stop_and_join(self, timeout: float = 10.0) -> None:
        """Sinaliza parada e aguarda o encerramento da thread.

        Args:
            timeout: Segundos máximos de espera.
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        logger.info("OCRWorker encerrado")

    def _run(self) -> None:
        """Loop de OCR de dois estágios: localiza placa, lê texto, encaminha para db_queue."""
        while not self._stop_event.is_set() or not self._ocr_queue.empty():
            try:
                track_id, crop, event_meta = self._ocr_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # Estágio 1 — detecção de placa (quando PlateDetector disponível)
            ocr_crop = crop
            if self._plate_detector is not None:
                try:
                    plate_crop = self._plate_detector.detect(crop)
                    if plate_crop is not None and plate_crop.size > 0:
                        ocr_crop = plate_crop
                        logger.debug(
                            "PlateDetector: placa localizada para track_id=%d "
                            "(%dx%d px)",
                            track_id, plate_crop.shape[1], plate_crop.shape[0],
                        )
                    else:
                        logger.debug(
                            "PlateDetector: nenhuma placa detectada em track_id=%d "
                            "— usando crop do veículo como fallback",
                            track_id,
                        )
                except Exception:
                    logger.warning(
                        "PlateDetector: erro ao processar track_id=%d — fallback para crop do veículo",
                        track_id, exc_info=True,
                    )

            # Estágio 2 — leitura de texto com EasyOCR
            plate_text, plate_confidence = self._plate_ocr.read(ocr_crop)

            event_meta["plate_text"] = plate_text
            event_meta["plate_confidence"] = plate_confidence

            try:
                self._db_queue.put_nowait(event_meta)
            except queue.Full:
                logger.warning(
                    "db_queue cheia — evento de track_id=%d descartado", track_id
                )
