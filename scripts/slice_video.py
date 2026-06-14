"""
Corta os primeiros 90 segundos de data/inputs/BR232.mp4
e salva em data/inputs/video_cortado.mp4 usando apenas OpenCV.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2


def main() -> None:
    src = Path("data/inputs/BR232.mp4")
    dst = Path("data/inputs/video_cortado.mp4")
    target_seconds = 90

    if not src.exists():
        print(f"ERRO: arquivo de entrada não encontrado: {src}", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"ERRO: não foi possível abrir {src}", file=sys.stderr)
        sys.exit(1)

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    target_frames = int(fps * target_seconds)
    frames_to_write = min(target_frames, total_frames_src)

    print(f"Fonte       : {src}")
    print(f"Resolução   : {width}x{height}")
    print(f"FPS         : {fps:.3f}")
    print(f"Total src   : {total_frames_src} frames ({total_frames_src / fps:.1f}s)")
    print(f"Alvo        : {target_frames} frames ({target_seconds}s)")
    print(f"Será escrito: {frames_to_write} frames")
    print()

    dst.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dst), fourcc, fps, (width, height))

    log_interval = max(1, frames_to_write // 20)  # log a cada 5 %

    for i in range(frames_to_write):
        ok, frame = cap.read()
        if not ok:
            print(f"Stream encerrou inesperadamente no frame {i}.")
            break

        writer.write(frame)

        if (i + 1) % log_interval == 0 or (i + 1) == frames_to_write:
            pct = (i + 1) / frames_to_write * 100
            print(f"  Processando frame {i + 1:5d} de {frames_to_write} ({pct:.0f}%)")

    cap.release()
    writer.release()

    size_mb = dst.stat().st_size / (1024 * 1024)
    print(f"\nConcluido -> {dst}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
