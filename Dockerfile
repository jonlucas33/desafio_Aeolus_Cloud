# ── Base ────────────────────────────────────────────────────────────────────
# Python 3.11-slim mantém a imagem enxuta (~120 MB base) sem sacrificar
# compatibilidade com PyTorch/EasyOCR/Ultralytics.
FROM python:3.11-slim

# ── Dependências de sistema ──────────────────────────────────────────────────
# libgl1         : OpenCV (cv2) precisa de libGL.so.1 mesmo em modo headless
# libglib2.0-0   : dependência transitiva do OpenCV (GLib/GThread)
# libsm6 libxext6: requeridas por algumas operações internas do OpenCV
# libgomp1       : OpenMP — usado por operações paralelas do EasyOCR/PyTorch
# libxrender1    : requerida em alguns ambientes headless para cv2.imencode
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Ferramentas de build ─────────────────────────────────────────────────────
# setuptools e wheel devem existir ANTES de qualquer pip install que use
# um backend de build local (setuptools.backends.legacy). A imagem slim
# não garante versão suficientemente recente — forçamos o upgrade aqui.
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ── Código-fonte ─────────────────────────────────────────────────────────────
# COPY . . deve vir ANTES de pip install ".[dev]" porque o comando instala
# o pacote local (o ponto final) e precisa ler pyproject.toml + src/.
# Arquivos pesados (data/, models/, .git/, __pycache__/) são excluídos
# automaticamente pelo .dockerignore — sem lixo na imagem.
COPY . .

# ── Dependências Python ──────────────────────────────────────────────────────
RUN pip install --no-cache-dir ".[dev]"

# ── Diretórios de runtime ────────────────────────────────────────────────────
# Criados para o caso de o volume não ser montado (evitar FileNotFoundError).
RUN mkdir -p data/inputs data/outputs models

# ── Entrypoint ───────────────────────────────────────────────────────────────
RUN chmod +x /app/docker/entrypoint.sh
ENTRYPOINT ["/app/docker/entrypoint.sh"]
