"""
Loader de configuração Pydantic para o pipeline de contagem de veículos.

Responsabilidade única: carregar e validar config/settings.yaml,
expondo modelos tipados para cada seção de configuração.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class VideoSettings(BaseModel):
    """Configurações de entrada e saída de vídeo."""

    source: str
    output: str
    resize_width: int = Field(gt=0)


class ModelSettings(BaseModel):
    """Configurações do modelo de detecção YOLO."""

    weights: str
    confidence_threshold: float = Field(gt=0.0, lt=1.0)
    iou_threshold: float = Field(gt=0.0, lt=1.0)
    device: Literal["cuda", "cpu", "mps"]
    fp16: bool


class TrackingSettings(BaseModel):
    """Configurações do rastreador ByteTrack."""

    track_buffer: int = Field(gt=0)
    min_box_area: int = Field(ge=0)


class CountingSettings(BaseModel):
    """Configurações da lógica de cruzamento de linha virtual."""

    line_points: list[list[int]]
    direction: Literal["any", "top_to_bottom", "bottom_to_top"]
    min_displacement_px: int = Field(gt=0)
    class_vote_window: int = Field(gt=0)


class OCRSettings(BaseModel):
    """Configurações do módulo de OCR de placas."""

    enabled: bool
    min_bbox_area_ratio: float = Field(gt=0.0, lt=1.0)
    languages: list[str]


class DatabaseSettings(BaseModel):
    """Configurações de persistência no banco de dados."""

    backend: Literal["sqlite", "postgresql"]
    sqlite_path: str
    postgres_url: str


class RenderingSettings(BaseModel):
    """Configurações visuais do overlay desenhado sobre os frames."""

    line_color: list[int] = [0, 255, 255]   # BGR: amarelo
    line_thickness: int = 2


class Settings(BaseModel):
    """Configuração raiz do pipeline — agrega todas as seções."""

    video: VideoSettings
    model: ModelSettings
    tracking: TrackingSettings
    counting: CountingSettings
    ocr: OCRSettings
    database: DatabaseSettings
    rendering: RenderingSettings = RenderingSettings()


def load_settings(path: Path) -> Settings:
    """Carrega e valida o arquivo YAML de configuração.

    Args:
        path: Caminho para o arquivo settings.yaml.

    Returns:
        Instância validada de Settings.

    Raises:
        FileNotFoundError: Se o arquivo não existir no caminho fornecido.
        pydantic.ValidationError: Se algum campo falhar na validação.
    """
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return Settings.model_validate(raw)
