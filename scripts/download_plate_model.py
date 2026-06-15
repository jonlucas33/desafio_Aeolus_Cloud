"""
Script de download do modelo de detecção de placas veiculares.

Baixa o modelo Koushim/yolov8-license-plate-detection do HuggingFace Hub
usando huggingface_hub.hf_hub_download e salva em models/license_plate_detector.pt.
Idempotente: não baixa se o arquivo já existir.

Se o repositório exigir autenticação, passe o token via:
  - Variável de ambiente: HF_TOKEN=hf_...
  - Argumento CLI:        python scripts/download_plate_model.py --token hf_...
  - Login interativo:     huggingface-cli login
"""
from __future__ import annotations

from dotenv import load_dotenv
import os

load_dotenv() 
hf_token = os.getenv("HF_TOKEN")

import logging
import os
import shutil
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

_REPO_ID = "Koushim/yolov8-license-plate-detection"
_FILENAME = "best.pt"
_LOCAL_NAME = "license_plate_detector.pt"


def download_plate_model(
    dest_dir: Path = Path("models"),
    token: str | None = None,
) -> Path:
    """Baixa o modelo de detecção de placas do HuggingFace Hub.

    Usa hf_hub_download para obter o arquivo da cache local do HuggingFace,
    depois copia para dest_dir com o nome padronizado do projeto.

    Args:
        dest_dir: Diretório de destino (padrão: models/).
        token: Token de autenticação HuggingFace. Se None, usa HF_TOKEN do
               ambiente ou o token armazenado pelo `huggingface-cli login`.

    Returns:
        Caminho final do arquivo de peso.

    Raises:
        SystemExit: Em caso de falha no download.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / _LOCAL_NAME

    if dest.exists():
        logger.info(
            "Modelo já existe: %s (%.1f MB)", dest, dest.stat().st_size / 1e6
        )
        return dest

    # Resolve token: argumento explícito > variável de ambiente
    resolved_token = token or os.environ.get("HF_TOKEN")

    logger.info(
        "Baixando %s/%s do HuggingFace Hub (auth=%s)...",
        _REPO_ID, _FILENAME,
        "token" if resolved_token else "anônimo / cache de login",
    )

    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import RepositoryNotFoundError

        cached_path = hf_hub_download(
            repo_id=_REPO_ID,
            filename=_FILENAME,
            token=resolved_token,
        )
        logger.info("Arquivo baixado para cache: %s", cached_path)

        shutil.copy2(cached_path, dest)
        logger.info(
            "Modelo copiado para %s (%.1f MB)", dest, dest.stat().st_size / 1e6
        )

    except RepositoryNotFoundError:
        logger.error(
            "\n"
            "  Repositório não encontrado ou acesso negado (401).\n"
            "  O modelo '%s' pode exigir autenticação.\n\n"
            "  Soluções:\n"
            "    1. Faça login interativo:  ! huggingface-cli login\n"
            "    2. Passe o token via env:  $env:HF_TOKEN='hf_...'; python scripts/download_plate_model.py\n"
            "    3. Passe via argumento:    python scripts/download_plate_model.py --token hf_...\n\n"
            "  Obtenha seu token em https://huggingface.co/settings/tokens",
            _REPO_ID,
        )
        sys.exit(1)
    except Exception:
        logger.error(
            "Falha inesperada ao baixar o modelo de placas (%s/%s)",
            _REPO_ID, _FILENAME,
            exc_info=True,
        )
        sys.exit(1)

    return dest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            f"Download de '{_FILENAME}' de {_REPO_ID} → models/{_LOCAL_NAME}"
        )
    )
    parser.add_argument(
        "--dest", default="models", help="Diretório de destino (padrão: models/)"
    )
    parser.add_argument(
        "--token",
        default=None,
        help=(
            "Token de autenticação HuggingFace (ex: hf_...). "
            "Alternativa: defina a variável de ambiente HF_TOKEN."
        ),
    )
    args = parser.parse_args()

    path = download_plate_model(Path(args.dest), token=args.token)
    logger.info("Pronto: %s", path)
