# ARCHITECTURE.md — Referência Técnica do Sistema

**Projeto:** Vehicle Counter — Pipeline de Visão Computacional para contagem e classificação
de veículos em rodovias com OCR de placas e persistência em banco de dados.

**Versão da arquitetura:** 1.3
**Decisões validadas em:** Revisão conjunta Claude Sonnet + Gemini (ver seção § Decisões e Justificativas)

---

## Sumário

1. [Stack Técnica](#1-stack-técnica)
2. [Estrutura de Diretórios](#2-estrutura-de-diretórios)
3. [Pipeline de Processamento](#3-pipeline-de-processamento)
4. [Módulos — Especificação Detalhada](#4-módulos--especificação-detalhada)
5. [Lógica de Negócio — Crossing Logic](#5-lógica-de-negócio--crossing-logic)
6. [Schema do Banco de Dados](#6-schema-do-banco-de-dados)
7. [Dockerfile e Conteinerização](#7-dockerfile-e-conteinerização)
8. [Decisões Arquiteturais e Justificativas](#8-decisões-arquiteturais-e-justificativas)
9. [Riscos e Mitigações](#9-riscos-e-mitigações)

---

## 1. Stack Técnica

| Componente | Tecnologia | Versão mínima | Justificativa |
|---|---|---|---|
| Linguagem | Python | 3.11 | `tomllib` nativo, type hints melhorados |
| Detecção + Classificação | YOLOv8s (Ultralytics) | 8.0 | Melhor balanço mAP/FPS; COCO inclui car, truck, bus; ecossistema maduro |
| Fallback leve (CPU) | YOLOv8n | 8.0 | Flag `model: yolov8n.pt` no settings.yaml |
| Tracking | ByteTrack (via Ultralytics) | — | IoU-based, sem ReID neural, mantém FPS alto; `track_buffer` configurável |
| OCR (primary) | fast-alpr (ONNX) | 0.2+ | Pipeline end-to-end especializado em placas; detector + OCR em um único modelo ONNX |
| OCR (fallback) | EasyOCR | 1.7+ | Selecionável via `ocr.engine: "easyocr"` no settings.yaml |
| Validação de placa | Regex Python | — | Mercosul `[A-Z]{3}[0-9][A-Z][0-9]{2}` + padrão antigo `[A-Z]{3}[0-9]{4}` |
| ORM / Banco | SQLAlchemy + SQLite | 2.0 / 3.x | SQLite por padrão; flag `DB_BACKEND=postgresql` sem mudança de código |
| Vídeo I/O | OpenCV Headless | 4.8+ | `opencv-python-headless` — sem dependências de display para Docker |
| Configuração | PyYAML + Pydantic | — | `settings.yaml` validado em runtime com Pydantic BaseSettings |
| Testes | pytest | 7+ | + `pytest-mock` para mock de câmera/GPU |

### Por que YOLOv8s e não YOLOv11s

YOLOv11s ainda apresenta instabilidade no ecossistema de exportação (TensorRT, ONNX) e
na integração com ByteTrack via Ultralytics. YOLOv8s tem benchmarks extensivos publicados,
maior comunidade e comportamento previsível em produção. A migração para v11 pode ser feita
trocando apenas `model.weights` no `settings.yaml` quando o ecossistema estabilizar.

### Por que ByteTrack e não DeepSORT

DeepSORT exige uma rede neural de Re-Identificação (ReID) rodando por frame para extrair
embeddings de aparência — custo adicional por frame na GPU. ByteTrack usa apenas
IoU geométrico em dois rounds (detecções de alta e baixa confiança), sendo mais leve.
Para câmera fixa de rodovia sem oclusão severa, ByteTrack é suficiente. BoT-SORT fica como
pivot documentado se a oclusão do vídeo de entrada exigir ReID.

---

## 2. Estrutura de Diretórios

```
vehicle-counter/
│
├── CLAUDE.md                  # Instruções operacionais para Claude Code
├── ARCHITECTURE.md            # Este arquivo
├── TASKS.md                   # Backlog faseado de implementação
├── README.md                  # Documentação pública do projeto
│
├── config/
│   └── settings.yaml          # Todos os parâmetros ajustáveis
│
├── src/
│   ├── domain.py              # Dataclasses canônicas: Detection, Track, VehicleEvent
│   ├── config.py              # Loader Pydantic do settings.yaml
│   │
│   ├── capture/
│   │   └── video_capture.py   # Thread Producer: lê frames → frame_queue
│   │
│   ├── detection/
│   │   └── yolo_detector.py   # Wrapper YOLOv8: frame → List[Detection]
│   │
│   ├── tracking/
│   │   └── bytetrack_wrapper.py  # Wrapper ByteTrack: List[Detection] → List[Track]
│   │
│   ├── counting/
│   │   └── crossing_logic.py  # CrossingCounter: produto vetorial + votação majoritária
│   │
│   ├── ocr/
│   │   ├── plate_ocr.py           # OCRWorker (EasyOCR) + PlateOCR + _validate_plate
│   │   ├── plate_detector.py      # PlateDetector: localiza placa em crop de veículo (YOLOv8)
│   │   └── fast_alpr_worker.py    # FastAlprWorker: OCR end-to-end via fast-alpr ONNX
│   │
│   ├── database/
│   │   ├── models.py          # SQLAlchemy ORM: VehicleEvent table
│   │   └── db_writer.py       # Thread Consumer: db_queue → bulk insert
│   │
│   ├── rendering/
│   │   └── overlay_renderer.py  # Desenha bbox, linha virtual, contador no frame
│   │
│   └── main.py                # Ponto de entrada: monta e inicia o pipeline completo
│
├── models/                    # Pesos .pt e .onnx (gitignored — baixar via scripts)
│   └── .gitkeep
│
├── data/
│   ├── inputs/                # Vídeo original (gitignored)
│   └── outputs/               # Vídeo anotado + events.db (gitignored)
│
├── tests/
│   ├── unit/
│   │   ├── test_crossing_logic.py
│   │   ├── test_yolo_detector.py
│   │   ├── test_plate_ocr.py
│   │   ├── test_plate_detector.py
│   │   └── test_fast_alpr_worker.py
│   └── integration/
│       └── test_pipeline.py
│
├── scripts/
│   ├── download_models.py       # Baixa yolov8n.pt e yolov8s.pt via Ultralytics
│   ├── download_plate_model.py  # Baixa license_plate_detector.pt via HuggingFace Hub
│   ├── benchmark_detection.py   # Benchmark YOLOv8n vs YOLOv8s (FPS, crossings)
│   └── benchmark_ocr.py         # Benchmark EasyOCR vs fast-alpr (placas, FPS)
│
├── docker/
│   ├── Dockerfile
│   └── entrypoint.sh          # Baixa modelos se ausentes, depois executa main.py
│
├── docker-compose.yml
├── pyproject.toml             # Dependências e metadados do projeto
├── .gitignore
└── .env.example               # Variáveis de ambiente documentadas
```

---

## 3. Pipeline de Processamento

### Visão geral das threads

```
┌─────────────────────────────────────────────────────────────────┐
│  Thread: VideoCapture (Producer)                                │
│  cv2.VideoCapture → frame_queue (maxsize=3)                     │
└────────────────────────────┬────────────────────────────────────┘
                             │ frame
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Thread Principal: Inference Loop (Consumer + Producer)         │
│                                                                 │
│  1. frame = frame_queue.get()                                   │
│  2. detections = YoloDetector.detect(frame)                     │
│  3. tracks = ByteTrackWrapper.update(detections)                │
│  4. crossed_ids = CrossingCounter.update(tracks, frame)         │
│     ├─ Se crossed: ocr_queue.put(crop, track_id)  (não bloqueia)│
│     └─ db_queue.put(VehicleEvent)                 (não bloqueia)│
│  5. annotated = OverlayRenderer.draw(frame, tracks, counter)    │
│  6. video_writer.write(annotated)                               │
└──────────────┬──────────────────────────┬───────────────────────┘
               │                          │
               ▼                          ▼
┌──────────────────────────────────┐   ┌──────────────────────────────────────┐
│  Thread: OCR Worker              │   │  Thread: DB Writer (Consumer)        │
│  FastAlprWorker (padrão)         │   │                                      │
│  ou OCRWorker (engine=easyocr)   │   │  db_queue → batch de 10 eventos      │
│                                  │   │  → session.bulk_save_objects()       │
│  fast-alpr: ONNX end-to-end      │   │  → session.commit()                  │
│  EasyOCR: CLAHE + binarização    │   │                                      │
│  → regex validation              │   │  Flush forçado a cada 5s             │
│  → db_queue.put_nowait()         │   │  (garante persistência sem acúmulo)  │
└──────────────────────────────────┘   └──────────────────────────────────────┘
```

### Controle de filas

| Fila | `maxsize` | Comportamento quando cheia |
|---|---|---|
| `frame_queue` | 3 | Producer descarta frame mais antigo (frame drop intencional) |
| `ocr_queue` | **10** | `put_nowait()` — descarta crop, loga `DEBUG`; evita CPU starvation do ONNX |
| `db_queue` | 200 | `put_nowait()` — loga `WARNING` se cheia (evento perdido é crítico) |

### Pré-processamento de frame

```python
# Antes de passar ao YOLO
frame_resized = letterbox(frame, new_shape=1280)  # mantém aspect ratio
# YOLOv8 aceita BGR diretamente — não converter para RGB
# FP16 ativado via model.half() se device == "cuda"
```

---

## 4. Módulos — Especificação Detalhada

### 4.1 `src/capture/video_capture.py`

**Responsabilidade:** Ler frames de forma não-bloqueante e manter `frame_queue` preenchida.

```python
class VideoCapture:
    def __init__(self, source: str | int, queue: Queue, settings: CaptureSettings): ...
    def start(self) -> None: ...   # inicia thread daemon
    def stop(self) -> None: ...
    def is_alive(self) -> bool: ...
```

Implementação da thread: `while not self._stop_event.is_set()`. Se a fila estiver cheia,
descartar o frame mais antigo com `queue.get_nowait()` antes de inserir o novo (frame drop
deliberado para nunca acumular latência).

### 4.2 `src/detection/yolo_detector.py`

**Responsabilidade:** Encapsular o modelo YOLOv8 e retornar `List[Detection]`.

```python
class YoloDetector:
    def __init__(self, settings: ModelSettings): ...
    def detect(self, frame: np.ndarray) -> list[Detection]: ...
    def warmup(self, frame_shape: tuple[int, int, int], n_frames: int = 3) -> None: ...  # GPU warmup antes do loop
```

Filtrar apenas classes relevantes: `[car=2, motorcycle=3, bus=5, truck=7]` do COCO.
`warmup(frame_shape)` deve ser chamado em `main.py` antes de iniciar o loop,
passando o shape real do vídeo lido via `VideoCapture`. O frame sintético deve
passar pelo mesmo pipeline de pré-processamento (letterbox + normalização) para
garantir que a GPU aloque os tensores no shape correto. Shape diferente no warmup
causa realocação no primeiro frame real, zerando o benefício do aquecimento.

### 4.3 `src/tracking/bytetrack_wrapper.py`

**Responsabilidade:** Manter identidade de objetos entre frames.

```python
class ByteTrackWrapper:
    def __init__(self, settings: TrackingSettings): ...
    def update(self, detections: list[Detection], frame: np.ndarray) -> list[Track]: ...
    def reset(self) -> None: ...
```

Usar `model.track()` da Ultralytics com `tracker="bytetrack.yaml"` ou instanciar
`supervision.ByteTrack` diretamente. Calcular `centroid` automaticamente a partir de `bbox_xyxy`
antes de retornar os `Track` objects.

### 4.4 `src/counting/crossing_logic.py`

Ver seção completa § 5.

### 4.5 `src/ocr/` — Módulos de OCR

O engine OCR é selecionável via `settings.ocr.engine` (`"fast_alpr"` | `"easyocr"`).

**`fast_alpr_worker.py` — FastAlprWorker (engine padrão, v1.3.0)**

Thread consumidora que usa `fast-alpr` (ONNX end-to-end). Processa um crop de veículo
por vez: detector YOLO localiza a placa dentro do crop → OCR `cct-xs-v2-global-model`
lê o texto → `_validate_plate()` aplica regex Mercosul/Antigo.

**`plate_ocr.py` — OCRWorker + PlateOCR (engine alternativo)**

Thread consumidora usando EasyOCR. O `PlateDetector` (estágio 1) localiza a placa no
crop via YOLOv8 (`models/license_plate_detector.pt`); o `PlateOCR` (estágio 2) aplica
CLAHE + binarização + EasyOCR. Fallback para crop completo se o detector não encontrar placa.

**`plate_detector.py` — PlateDetector**

Wrapper YOLOv8 para localização de placa dentro de crop de veículo. Se a altura do crop
for menor que 64px, aplica upscale cúbico antes da inferência e remapeia as coordenadas.
Retorna sub-crop com margem de 4px ou `None` se nenhuma placa for detectada.

**Estratégia de best-crop (CrossingCounter → OCR):**

O OCR dispara uma única vez por `track_id` usando o crop de maior área de bbox histórico.
Implementado no `CrossingCounter` via `_best_crop[track_id]` e `_max_bbox_area[track_id]`.
No evento de cruzamento: `ocr_queue.put_nowait((track_id, best_crop, event_meta))`.

### 4.6 `src/database/models.py`

Ver seção § 6.

### 4.7 `src/rendering/overlay_renderer.py`

**Responsabilidade:** Desenhar anotações no frame sem modificar o frame original.

```python
class OverlayRenderer:
    def __init__(self, settings: RenderSettings, line_points: list): ...
    def draw(self, frame: np.ndarray, tracks: list[Track],
             counter: int, fps: float) -> np.ndarray: ...
```

Sempre trabalhar em cópia do frame: `annotated = frame.copy()`.
Desenhar: linha virtual colorida, bbox com cor por classe, track_id, classe, FPS no canto,
contador total em destaque.

---

## 5. Lógica de Negócio — Crossing Logic

### Produto vetorial 2D para detecção de cruzamento

Dados:
- Linha virtual `Lv` definida pelos pontos `A(x1, y1)` e `B(x2, y2)`
- Centróide do veículo no frame anterior: `P_prev(px, py)`
- Centróide do veículo no frame atual: `P_curr(cx, cy)`

A detecção de cruzamento usa o sinal do produto vetorial (cross product 2D) para determinar
de qual lado da linha cada ponto está:

```python
def _side(ax, ay, bx, by, px, py) -> float:
    """Retorna positivo se P está à esquerda de AB, negativo à direita."""
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)

def _crossed(line_a, line_b, p_prev, p_curr) -> bool:
    d1 = _side(*line_a, *line_b, *p_prev)
    d2 = _side(*line_a, *line_b, *p_curr)
    # Cobre o caso em que o centróide cai exatamente sobre a linha (d==0).
    # (d1 > 0) != (d2 > 0) perde esse caso por causa de -0.0 em IEEE 754.
    return d1 * d2 < 0 or (d1 == 0) != (d2 == 0)
```

> **Nota de implementação:** A condição `d1 * d2 < 0 or (d1 == 0) != (d2 == 0)`
> protege contra o comportamento de `-0.0` em IEEE 754, que faria a versão naïve
> `(d1 > 0) != (d2 > 0)` perder silenciosamente cruzamentos exatos sobre a linha.

### Filtro de jitter e direção

```python
def _is_valid_movement(p_prev, p_curr, min_displacement_px) -> bool:
    dx = p_curr[0] - p_prev[0]
    dy = p_curr[1] - p_prev[1]
    return (dx**2 + dy**2) ** 0.5 >= min_displacement_px

def _is_correct_direction(p_prev, p_curr, direction: str) -> bool:
    if direction == "any":
        return True
    dy = p_curr[1] - p_prev[1]
    if direction == "top_to_bottom":
        return dy > 0
    if direction == "bottom_to_top":
        return dy < 0
    return True
```

### Interface completa de CrossingCounter

```python
class CrossingCounter:
    def __init__(self, line_points: list, direction: str,
                 min_displacement_px: int, class_vote_window: int): ...

    def update(self, tracks: list[Track], frame: np.ndarray) -> list[int]:
        """Retorna lista de track_ids que cruzaram neste frame.
        O parâmetro frame é necessário para capturar _best_crop[track_id]."""
        ...

    def get_vehicle_class(self, track_id: int) -> str:
        """Retorna classe majoritária dos últimos N frames para o track_id."""
        ...

    @property
    def count(self) -> int:
        """Total de veículos contados na sessão."""
        ...
```

**Estruturas de estado internas:**
- `_crossed_ids: set[int]` — garante idempotência (um ID nunca é contado duas vezes)
- `_previous_centroids: dict[int, tuple]` — centróide do frame anterior por track_id
- `_class_votes: dict[int, Counter]` — Counter de votos de classe por track_id
- `_best_crop: dict[int, np.ndarray]` — melhor crop (maior bbox) capturado por track_id
- `_max_bbox_area: dict[int, float]` — área máxima de bbox vista por track_id (substitui ao crescer)

### Mapeamento de classe COCO → categoria de negócio

```python
COCO_TO_VEHICLE_CLASS = {
    2: "car",        # será refinado por heurística de tamanho
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

def _refine_class(class_id: int, bbox_area: float, frame_area: float, aspect_ratio: float) -> str:
    """Heurística de área/aspecto para diferenciar SUV/Picape de Sedan/Hatch.

    Thresholds configuráveis via settings.yaml:
      counting.suv_aspect_ratio_threshold (padrão 0.85)
      counting.truck_area_threshold (padrão 0.02)
    """
    base = COCO_TO_VEHICLE_CLASS.get(class_id, "unknown")
    if base == "car":
        ratio = bbox_area / frame_area
        return "suv_pickup" if ratio > 0.035 else "sedan_hatch"
    return base
```

---

## 6. Schema do Banco de Dados

```sql
CREATE TABLE vehicle_events (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id          INTEGER NOT NULL,
    vehicle_class     VARCHAR(50) NOT NULL,
    plate_text        VARCHAR(20),
    plate_confidence  REAL,            -- NÃO plate_conf
    frame_number      INTEGER NOT NULL,
    timestamp         DATETIME NOT NULL,  -- NÃO crossed_at
    session_id        VARCHAR(36) NOT NULL
);

CREATE INDEX idx_track_id  ON vehicle_events(track_id);
CREATE INDEX idx_session   ON vehicle_events(session_id);
```

`session_id` é gerado com `uuid.uuid4()` no início de `main.py` e passado para o `DbWriter`,
permitindo distinguir múltiplas execuções no mesmo banco SQLite sem apagar dados anteriores.

### SQLAlchemy Model

```python
class VehicleEvent(Base):
    __tablename__ = "vehicle_events"

    id                = Column(Integer, primary_key=True)
    track_id          = Column(Integer, nullable=False)
    vehicle_class     = Column(String(50), nullable=False)
    plate_text        = Column(String(20), nullable=True)
    plate_confidence  = Column(Float, nullable=True)   # NÃO plate_conf
    frame_number      = Column(Integer, nullable=False)
    timestamp         = Column(DateTime, nullable=False)  # NÃO crossed_at
    session_id        = Column(String(36), nullable=False)
```

### Configuração obrigatória da engine SQLite

O `DbWriter` roda em thread separada. O SQLite bloqueia acesso cross-thread por padrão.
A engine **deve** ser criada com os seguintes parâmetros — sem exceção:

```python
from sqlalchemy import create_engine, event

def create_sqlite_engine(path: str):
    engine = create_engine(
        f"sqlite:///{path}",
        connect_args={"check_same_thread": False},  # obrigatório para multithreading
    )

    @event.listens_for(engine, "connect")
    def set_wal_mode(dbapi_conn, _):
        dbapi_conn.execute("PRAGMA journal_mode=WAL")   # leituras concorrentes sem lock
        dbapi_conn.execute("PRAGMA synchronous=NORMAL") # performance sem risco de corrupção

    return engine
```

`WAL` (Write-Ahead Logging) permite que leituras (ex: consultas de debug) ocorram enquanto
o `DbWriter` está inserindo, eliminando o erro `database is locked` sob carga.

---

## 7. Dockerfile e Conteinerização

### Imagem base

```dockerfile
FROM python:3.11-slim
```

**Justificativa:** Imagem base mínima com Python 3.11. As dependências (PyTorch, Ultralytics,
fast-alpr, etc.) são instaladas via `pip install -r requirements.txt` no build. Mais
controlável e portável que `ultralytics/ultralytics:latest`, que pode mudar sem aviso e
inclui dependências desnecessárias para CPU. Para GPU, adicionar `--extra-index-url` do
PyTorch com CUDA.

### CPU-only (para máquinas sem GPU)

```dockerfile
FROM python:3.11-slim AS cpu
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Selecionado via `docker-compose --profile cpu up`.

### Pontos críticos do Dockerfile

```dockerfile
# opencv-python-headless: sem dependências X11 — obrigatório em servidores sem display
RUN pip install opencv-python-headless fast-alpr onnxruntime ...

# Modelos baixados no entrypoint, não na imagem (imagem menor, modelo atualizável)
COPY docker/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
```

### docker-compose.yml (estrutura)

```yaml
services:
  vehicle-counter:
    build: .
    volumes:
      - ./data:/app/data
      - ./models:/app/models    # persiste modelos entre builds
      - ./config:/app/config
    environment:
      - DB_BACKEND=sqlite
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## 8. Decisões Arquiteturais e Justificativas

| Decisão | Alternativa rejeitada | Motivo da escolha |
|---|---|---|
| Produto vetorial para crossing | Threshold em Y | Suporta linhas diagonais; robusto a câmeras oblíquas |
| Votação majoritária de classe | Classificação frame a frame | Elimina instabilidade; custo O(1) por frame |
| OCR no frame de bbox máxima | OCR em fila contínua | Melhor qualidade de crop; elimina processamento redundante |
| `python:3.11-slim` como base Docker | `ultralytics/ultralytics` | Imagem mínima e controlável; dependências explícitas em requirements.txt |
| SQLAlchemy como ORM | Queries SQL diretas | Abstração SQLite↔PostgreSQL via flag; sem mudança de código |
| Separação por domínio em `src/` | Estrutura flat (`core/`, `utils/`) | Cada módulo substituível e testável isoladamente |
| SQLAlchemy engine com `check_same_thread=False` + WAL | Engine padrão do SQLite | SQLite bloqueia acesso cross-thread por padrão; WAL elimina `database is locked` sob carga |
| Best-crop buffer para OCR trigger | Crop do frame de cruzamento | Linha virtual ≠ ponto de máxima proximidade; best-crop garante maior qualidade sem disparos redundantes |
| `warmup(frame_shape)` com shape real + pré-processamento | `warmup()` com `np.zeros((640,640,3))` | Shape diferente no warmup causa realocação de memória GPU no primeiro frame real |
| `frame_queue` maxsize=3 com frame drop | Buffer ilimitado | Controla latência; pipeline nunca acumula atraso |
| Dataclasses canônicas em `domain.py` | Dicts passados entre módulos | Contratos explícitos; type checking; sem acoplamento implícito |
| YOLOv8s como modelo de detecção | YOLOv8n | +14% cruzamentos (86→98); recall SUV/Picape melhor; benchmark medido em v1.3.0 |
| fast-alpr (ONNX) como engine OCR | EasyOCR | 49 vs 4 placas (50% vs 4.1%); conf 0.96 vs 0.57; FPS superior após fix (5.56 vs 4.78) |
| `ocr_queue` maxsize=10 | maxsize=50 | ONNX Runtime saturava cores CPU em background; maxsize=10 limita CPU starvation (1.11→5.56 FPS) |
| `put_nowait()` + `logger.debug` na ocr_queue | `put()` bloqueante | Loop principal nunca espera por OCR; descarte silencioso evita spam de WARNING |

---

## 9. Riscos e Mitigações

### Risco 1 — ID switching por oclusão severa

**Causa:** Veículo passa por baixo de caminhão ou placa de trânsito; ByteTrack perde
associação e reatribui novo `track_id`.

**Mitigação:**
- `track_buffer: 30` no ByteTrack — 30 frames de memória antes de considerar objeto perdido
- Set `_crossed_ids` garante que mesmo com novo ID, o veículo só é contado se cruzar a linha
  com o novo ID (o que é improvável se a linha estiver posicionada no meio da pista)
- Se oclusão for crítica no vídeo de entrada, ativar BoT-SORT como pivot: `tracker: "botsort.yaml"`

### Risco 2 — CPU starvation com OCR ONNX

**Causa:** O ONNX Runtime usa pool de threads próprio em C++ que, com filas grandes,
satura todos os cores da CPU, privando o PyTorch (YOLOv8) de tempo de execução.
Observado com fast-alpr e `ocr_queue` de `maxsize=50`: FPS caiu de 5.56 → 1.11.

**Mitigação:**
- `ocr_queue` com `maxsize=10` e `put_nowait()` — descarte silencioso (`logger.debug`)
- Disparo único por `track_id` no frame de bbox máxima
- Em GPU, ONNX usa `CUDAExecutionProvider` automaticamente — starvation não ocorre

### Risco 3 — Classificação incorreta SUV vs Sedan

**Causa:** YOLOv8 COCO classifica ambos como `car`; ângulo da câmera distorce proporções.

**Mitigação:**
- Votação majoritária: decision por Counter dos últimos 15 frames por `track_id`
- Heurística de área relativa: `bbox_area / frame_area > 0.035` → SUV/Picape
- Limiar de área ajustável via `settings.yaml` para calibrar por câmera específica
- Opção futura: fine-tuning de cabeçalho leve (EfficientNet-B0 congelado) com crops
  anotados extraídos do próprio vídeo de produção
