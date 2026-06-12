# CLAUDE.md — Instruções Operacionais para Claude Code

Este arquivo define como você deve se comportar ao trabalhar neste projeto.
Leia **ARCHITECTURE.md** antes de qualquer implementação.
Consulte **TASKS.md** para saber o que implementar e em qual ordem.

---

## Papel e Contexto

Você é o engenheiro de implementação de um pipeline de Visão Computacional para contagem
e classificação de veículos em rodovias. A arquitetura já foi definida e validada — seu papel
é **implementar fielmente o que está especificado**, propor melhorias pontuais quando
identificar um problema concreto, e nunca redesenhar módulos sem justificativa explícita.

---

## Regras Inegociáveis

### O que você NUNCA deve fazer

- **Nunca usar loop síncrono sequencial** (`read → infer → OCR → draw` em sequência no mesmo thread).
  Todo I/O de vídeo e OCR obrigatoriamente em threads separadas via `queue.Queue`.
- **Nunca rodar OCR em todos os frames.** OCR dispara uma única vez por `track_id`,
  usando o melhor crop já capturado — não o frame do cruzamento em si.
  Estratégia obrigatória: manter um `_best_crop` por `track_id` (substituído a cada frame
  em que `bbox_area > max_area_seen`). No evento de cruzamento, enviar `_best_crop` para
  a fila de OCR. Isso garante o crop de maior qualidade sem disparos redundantes.
  **Nunca** usar o crop do frame exato do cruzamento: a linha virtual geralmente não coincide
  com o ponto de máxima proximidade com a câmera.
- **Nunca usar threshold simples em Y** para detectar cruzamento de linha virtual.
  Use obrigatoriamente produto vetorial 2D (cross product). Ver `ARCHITECTURE.md § Crossing Logic`.
- **Nunca usar `opencv-python`** nas dependências. Sempre `opencv-python-headless`.
- **Nunca criar estado global.** Toda lógica encapsulada em classes com injeção de dependência.
- **Nunca criar a engine SQLAlchemy sem `connect_args={"check_same_thread": False}`** ao usar SQLite.
  Sem isso, qualquer acesso ao banco a partir do `DbWriter` (thread separada) lança
  `ProgrammingError` em runtime. Obrigatório também ativar WAL mode via pragma (ver `ARCHITECTURE.md § 6`).
- **Nunca commitar arquivos `.pt` de modelo**, vídeos de input ou o arquivo `.db` do SQLite.
  Eles estão no `.gitignore`.
- **Nunca fazer warmup da GPU com frame sintético de shape arbitrário.**
  O método `warmup()` deve criar o frame com as dimensões exatas do vídeo de entrada
  (`VideoCapture.frame_width`, `frame_height`) e passar pelo mesmo pipeline de pré-processamento
  (letterbox + normalização) do loop principal. Shape diferente no warmup causa realocação de
  memória da GPU no primeiro frame real, eliminando completamente o benefício do warmup.
- **Nunca quebrar a interface pública de um módulo já implementado e testado** sem atualizar
  todos os seus consumidores no mesmo commit.

### O que você SEMPRE deve fazer

- **Sempre ler `TASKS.md`** antes de começar qualquer sessão e marcar a tarefa como `[x]` ao concluir.
- **Sempre escrever type hints completos** em todas as funções e métodos.
- **Sempre usar dataclasses** para estruturas de dados de domínio (`Track`, `VehicleEvent`, etc.).
- **Sempre preferir `pathlib.Path`** em vez de strings para caminhos de arquivo.
- **Sempre configurar via `config/settings.yaml`**, nunca via constantes hardcoded no código.
- **Sempre rodar `pytest tests/`** antes de declarar uma tarefa concluída.
- **Sempre gerar docstrings no padrão Google** em classes e métodos públicos.

---

## Convenções de Código

### Estrutura obrigatória de cada módulo

```python
# src/counting/crossing_logic.py
"""
Módulo de lógica de cruzamento de linha virtual.

Responsabilidade única: determinar quais track_ids cruzaram a linha
neste frame e manter histórico de votos de classificação.
"""
from __future__ import annotations
# imports stdlib
# imports third-party
# imports internos (src.*)
```

### Nomenclatura

| Tipo | Convenção | Exemplo |
|---|---|---|
| Classes | PascalCase | `CrossingCounter`, `ByteTrackWrapper` |
| Funções/métodos | snake_case | `update_tracks`, `get_vehicle_class` |
| Constantes de config | UPPER_SNAKE | nunca — use `settings.yaml` |
| Arquivos de módulo | snake_case | `crossing_logic.py` |
| Variáveis privadas | `_prefixo` | `_crossed_ids`, `_class_votes` |

### Padrão de logging

```python
import logging
logger = logging.getLogger(__name__)
# Use: logger.debug / logger.info / logger.warning / logger.error
# Nunca use print() em código de produção
```

### Tratamento de erros

- Erros de hardware (câmera, GPU): logar como `ERROR` e encerrar com `sys.exit(1)` e mensagem clara.
- Erros de OCR/DB individuais: logar como `WARNING` e continuar — nunca travar o pipeline principal.
- Frames corrompidos: descartar silenciosamente com `logger.debug`.

---

## Ordem de Implementação (fases)

Implemente **estritamente nesta ordem**. Não avance para a próxima fase sem testes passando.

```
Fase 1 — Fundação
  1.1  Estrutura de diretórios + setup do projeto (pyproject.toml, .gitignore)
  1.2  config/settings.yaml + src/config.py (loader Pydantic)
  1.3  src/capture/video_capture.py (thread Producer)
  1.4  src/detection/yolo_detector.py (wrapper YOLOv8)
  1.5  src/tracking/bytetrack_wrapper.py
  1.6  Teste de integração: captura → detecção → tracking com vídeo de amostra

Fase 2 — Lógica de Negócio
  2.1  src/counting/crossing_logic.py (produto vetorial + votação)
  2.2  src/rendering/overlay_renderer.py (linha virtual, bbox, contador)
  2.3  Pipeline principal: src/main.py (integração das fases 1 e 2)
  2.4  Teste end-to-end com saída de vídeo anotado

Fase 3 — Diferenciais
  3.1  src/ocr/plate_ocr.py (crop por bbox máxima + PaddleOCR + regex)
  3.2  src/database/models.py (SQLAlchemy ORM — VehicleEvent)
  3.3  src/database/db_writer.py (thread Consumer da db_queue)
  3.4  Integração OCR + DB no pipeline principal
  3.5  Teste de persistência e validação de dados

Fase 4 — Produção
  4.1  Dockerfile + docker-compose.yml
  4.2  README.md completo
  4.3  Testes de carga (benchmark de FPS com e sem OCR)
```

---

## Fluxo de Comunicação Entre Módulos

```
[VideoCapture Thread]
    │  frame_queue (Queue, maxsize=3)
    ▼
[Main Thread: Inference Loop]
    │  YoloDetector.detect(frame) → List[Detection]
    │  ByteTrackWrapper.update(detections) → List[Track]
    │  CrossingCounter.update(tracks) → List[crossed_track_ids]
    │  ├─ ocr_queue (Queue, maxsize=50) → [OCR Thread Pool]
    │  └─ db_queue  (Queue, maxsize=200) → [DB Writer Thread]
    │  OverlayRenderer.draw(frame, tracks, counter)
    ▼
[VideoWriter / cv2.imshow]
```

Nenhum módulo deve conhecer o módulo que está acima ou abaixo dele na cadeia,
apenas as interfaces de dados trocadas (dataclasses definidas em `src/domain.py`).

---

## Dataclasses Canônicas (src/domain.py)

Estes são os contratos de dados entre módulos. **Nunca altere sem atualizar todos os consumidores.**

```python
@dataclass
class Detection:
    bbox_xyxy: np.ndarray   # [x1, y1, x2, y2] float32
    confidence: float
    class_id: int
    class_name: str

@dataclass
class Track:
    track_id: int
    bbox_xyxy: np.ndarray
    confidence: float
    class_id: int
    class_name: str
    centroid: tuple[float, float]   # (cx, cy) calculado automaticamente

@dataclass
class VehicleEvent:
    track_id: int
    vehicle_class: str
    plate_text: str | None
    plate_confidence: float | None
    frame_number: int
    timestamp: datetime
```

---

## Configuração (config/settings.yaml)

Todo parâmetro ajustável deve existir aqui. Nunca hardcode.

```yaml
video:
  source: "data/inputs/video.mp4"   # caminho ou índice de câmera (0, 1...)
  output: "data/outputs/result.mp4"
  resize_width: 1280

model:
  weights: "models/yolov8s.pt"
  confidence_threshold: 0.45
  iou_threshold: 0.5
  device: "cuda"                    # "cuda" | "cpu" | "mps"
  fp16: true

tracking:
  track_buffer: 30                  # frames de memória do ByteTrack
  min_box_area: 100

counting:
  line_points: [[0, 540], [1280, 540]]   # [[x1,y1],[x2,y2]]
  direction: "top_to_bottom"             # "any" | "top_to_bottom" | "bottom_to_top"
  min_displacement_px: 5
  class_vote_window: 15

ocr:
  enabled: true
  min_bbox_area_ratio: 0.01         # só tenta OCR se bbox > 1% do frame
  languages: ["pt", "en"]

database:
  backend: "sqlite"                 # "sqlite" | "postgresql"
  sqlite_path: "data/outputs/events.db"
  postgres_url: ""                  # via env: DATABASE_URL
```

---

## Sobre Modelos e Arquivos Grandes

- O arquivo de pesos `models/yolov8s.pt` **não é commitado**. Baixe com:
  ```bash
  python scripts/download_models.py
  ```
- O script usa a API da Ultralytics e verifica hash SHA256 após download.
- Para ambiente Docker, os modelos são baixados no `docker-compose up` via `entrypoint.sh`
  se não existirem no volume montado.

---

## Quando Encontrar Ambiguidade

Se uma tarefa do `TASKS.md` não estiver suficientemente especificada, **pergunte antes de implementar**.
Descreva o que está ambíguo e proponha duas ou três abordagens com trade-offs.
Não assuma e implemente silenciosamente.
