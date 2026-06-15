#!/bin/bash
# Entrypoint do container vehicle-counter.
# Verifica o modelo de detecção e inicia o pipeline.
set -e

MODEL_PATH="${MODEL_PATH:-models/yolov8n.pt}"

# Se o modelo não estiver no volume montado, baixa via Ultralytics.
# A Ultralytics armazena automaticamente em $HOME/.config/Ultralytics/ como
# fallback; aqui forçamos o path do projeto para persistência via volume.
if [ ! -f "$MODEL_PATH" ]; then
    echo "[entrypoint] Modelo não encontrado em $MODEL_PATH — baixando..."
    python scripts/download_models.py
else
    echo "[entrypoint] Modelo encontrado: $MODEL_PATH"
fi

echo "[entrypoint] Iniciando pipeline..."
exec python main.py "$@"
