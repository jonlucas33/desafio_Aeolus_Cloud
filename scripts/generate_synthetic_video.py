"""
Gera vídeo sintético para validação E2E do pipeline.

Cria data/inputs/video.mp4 com 3 retângulos coloridos descendo verticalmente,
cruzando a linha virtual em y=540, simulando veículos em rodovia.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    out_path = Path("data/inputs/video.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    W, H = 1280, 720
    FPS = 30
    DURATION_S = 10
    N_FRAMES = FPS * DURATION_S

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, FPS, (W, H))

    # 3 "veículos": diferentes cores, posições X, tamanhos e velocidades
    vehicles = [
        {"x": 200, "y0": 50,  "w": 90, "h": 55, "speed": 6,  "color": (0, 255, 0)},   # verde
        {"x": 600, "y0": 0,   "w": 110, "h": 65, "speed": 4, "color": (0, 0, 255)},    # vermelho
        {"x": 1000,"y0": 80,  "w": 80, "h": 50, "speed": 8,  "color": (255, 165, 0)},  # laranja
    ]

    # Fundo cinza-escuro simulando asfalto
    background = np.full((H, W, 3), (40, 40, 40), dtype=np.uint8)

    # Faixas claras
    for lane_x in [427, 854]:
        cv2.line(background, (lane_x, 0), (lane_x, H), (80, 80, 80), 3)

    for frame_idx in range(N_FRAMES):
        frame = background.copy()

        for v in vehicles:
            cy = int(v["y0"] + frame_idx * v["speed"]) % (H + 200) - 100
            x1, y1 = v["x"], cy
            x2, y2 = v["x"] + v["w"], cy + v["h"]
            if y2 > 0 and y1 < H:
                cv2.rectangle(frame, (x1, max(0, y1)), (x2, min(H, y2)), v["color"], -1)

        # Linha virtual em y=540
        cv2.line(frame, (0, 540), (W, 540), (0, 255, 255), 1)

        # Frame counter
        cv2.putText(frame, f"frame {frame_idx:04d}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        writer.write(frame)

    writer.release()
    print(f"Vídeo gerado: {out_path} ({N_FRAMES} frames, {DURATION_S}s a {FPS}fps)")


if __name__ == "__main__":
    main()
