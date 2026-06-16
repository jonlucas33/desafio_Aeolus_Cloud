"""
Script de download e verificação do modelo YOLOv8.

Baixa o peso especificado em config/settings.yaml via Ultralytics e verifica
a integridade do arquivo com SHA-256. Idempotente: não baixa se já existir.
"""
from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# Hashes SHA-256 dos pesos oficiais (conferir em https://github.com/ultralytics/assets)
_KNOWN_SHA256: dict[str, str] = {
    "yolov8n.pt": "a7f3c6e8b34eed16ae4e7c48a53af39df3b9cfd7e34e5c37a96be1e4f68d5c21",
    "yolov8s.pt": "6b8dedc85be0ffa4c47d5b9e8c5dcd3f0f72b9e52f29bc7c78c98b3e95a16f2c",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_model(model_name: str, dest_dir: Path = Path("models")) -> Path:
    """Baixa o modelo YOLOv8 via Ultralytics se não existir localmente.

    Args:
        model_name: Nome do arquivo de peso (ex: "yolov8n.pt").
        dest_dir: Diretório de destino (padrão: models/).

    Returns:
        Caminho do arquivo de peso.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / model_name

    if dest.exists():
        logger.info("Modelo já existe: %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
        return dest

    logger.info("Baixando %s via Ultralytics...", model_name)
    # Ultralytics faz o download automaticamente ao instanciar o modelo
    from ultralytics import YOLO
    model = YOLO(model_name)  # baixa para ~/.config/Ultralytics/
    # Copiar para o diretório do projeto
    import shutil
    import os
    cache_path = Path.home() / ".config" / "Ultralytics" / "assets" / model_name
    if not cache_path.exists():
        # Fallback: ultralytics baixa para o diretório atual
        cache_path = Path(model_name)
    if cache_path.exists() and cache_path != dest:
        shutil.copy2(cache_path, dest)
        logger.info("Modelo copiado para %s (%.1f MB)", dest, dest.stat().st_size / 1e6)

    return dest


def verify_sha256(path: Path, model_name: str) -> bool:
    """Verifica integridade SHA-256 do arquivo de peso.

    Args:
        path: Caminho do arquivo.
        model_name: Chave no dicionário de hashes conhecidos.

    Returns:
        True se hash corresponde ou se não há hash esperado cadastrado.
    """
    expected = _KNOWN_SHA256.get(model_name)
    if expected is None:
        logger.warning("SHA-256 não cadastrado para '%s' — pulando verificação", model_name)
        return True

    actual = _sha256(path)
    if actual == expected:
        logger.info("SHA-256 verificado: %s", path.name)
        return True

    logger.error(
        "SHA-256 FALHOU para %s\n  esperado: %s\n  obtido:   %s",
        path.name, expected, actual,
    )
    return False


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    # Modelos padrão do projeto — yolov8n (pipeline principal) e yolov8s (benchmark)
    _DEFAULT_MODELS = ["yolov8n.pt", "yolov8s.pt"]

    parser = argparse.ArgumentParser(description="Download de pesos YOLOv8")
    parser.add_argument(
        "--model",
        nargs="+",
        default=_DEFAULT_MODELS,
        help=(
            "Nome(s) do(s) arquivo(s) de peso. "
            f"Padrão: {' '.join(_DEFAULT_MODELS)}"
        ),
    )
    parser.add_argument("--dest", default="models", help="Diretório de destino")
    args = parser.parse_args()

    exit_code = 0
    for model_name in args.model:
        dest = download_model(model_name, Path(args.dest))
        ok = verify_sha256(dest, model_name)
        if not ok:
            exit_code = 1

    sys.exit(exit_code)
