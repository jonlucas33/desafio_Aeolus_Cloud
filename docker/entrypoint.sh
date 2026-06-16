#!/bin/bash
set -e

# Download YOLOv8s if not present (default model for v1.3.0)
if [ ! -f "models/yolov8s.pt" ]; then
    echo "[entrypoint] Downloading YOLOv8s model..."
    python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"
    mv yolov8s.pt models/ 2>/dev/null || true
fi

# fast-alpr models are downloaded automatically on first use
# to ~/.cache/open-image-models and ~/.cache/fast-plate-ocr

echo "[entrypoint] Starting pipeline..."
exec python main.py --config config/settings.yaml "$@"
