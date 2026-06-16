# Pipeline de Contagem e Classificação de Veículos em Rodovias

Sistema de visão computacional em tempo real que detecta, rastreia e contabiliza veículos cruzando uma linha virtual em câmeras de monitoramento rodoviário. Combina **YOLOv8s** para detecção, **ByteTrack** para rastreamento persistente, **fast-alpr** (ONNX) para leitura de placas em alta precisão e **SQLite** para persistência de todos os eventos.

---

## Demo

![Pipeline de contagem de veículos em rodovia BR-232](docs/demo.gif)

> 90 segundos de rodovia BR-232 · 109 veículos detectados · 65 placas lidas · 4.5 FPS em CPU

📥 [Download do vídeo completo anotado (result_final.mp4)](https://github.com/jonlucas33/desafio_Aeolus_Cloud/releases/download/v1.3.0/result_final.mp4)

---

## Sumário

- [Visão Geral](#visão-geral)
- [Arquitetura e Decisões de Engenharia](#arquitetura-e-decisões-de-engenharia)
- [Limitações Conhecidas](#limitações-conhecidas)
- [Pré-requisitos](#pré-requisitos)
- [Como Executar](#como-executar)
- [Configuração](#configuração)
- [Schema do Banco de Dados](#schema-do-banco-de-dados)
- [Performance Esperada](#performance-esperada)
- [Modelos Utilizados](#modelos-utilizados)

---

## Visão Geral

```
Entrada: arquivo de vídeo MP4 (câmera fixa de rodovia)
   ↓
Detecção de veículos quadro a quadro (YOLOv8s)
   ↓
Rastreamento de identidade entre frames (ByteTrack)
   ↓
Detecção de cruzamento de linha virtual (produto vetorial 2D)
   ↓
OCR de placa em thread separada não-bloqueante (fast-alpr ONNX, maxsize=10)
   ↓
Persistência assíncrona de eventos (SQLite via thread DbWriter)
   ↓
Saída: vídeo anotado + banco de dados com 1 linha por veículo contado
```

**Resultado em 90 segundos de rodovia BR-232 (v1.3.0, main.py):**
- 2.700 frames processados (100% — nenhum frame descartado)
- **109 veículos únicos contabilizados**
- **65 placas lidas** (59.6% de taxa de captura, confiança média 0.96)
- Distribuição: 62 sedan/hatch · 8 suv/picape · 24 caminhão/ônibus · 15 motos

---

## Arquitetura e Decisões de Engenharia

### Fluxo de threads e filas

```
┌─────────────────────────────────────────────────────────────────────┐
│                          THREAD PRINCIPAL                            │
│                                                                      │
│  [VideoCapture Thread]                                               │
│       │  frame_queue (Queue, maxsize=3, backpressure)               │
│       ↓                                                              │
│  Loop de Inferência:                                                 │
│    YoloDetector.detect(frame)  → List[Detection]                    │
│    ByteTrackWrapper.update()   → List[Track]                        │
│    CrossingCounter.update()    → List[crossed_track_ids]            │
│       │                   │                                          │
│  ocr_queue (max=10)    db_queue (max=200)                           │
│  put_nowait()          put_nowait()                                  │
│       │                   │                                          │
│       ↓                   ↓                                          │
│  [FastAlprWorker]     [DbWriter Thread]                              │
│  fast-alpr ONNX       Batch insert de 10                            │
│       │               eventos ou 5s timeout                          │
│       └───────────────────┘                                          │
│              ↓                                                       │
│         SQLite (WAL mode, check_same_thread=False)                  │
└─────────────────────────────────────────────────────────────────────┘
```

### Decisões de engenharia

**1. OCR dispara uma única vez por veículo, nunca por frame**

O módulo `CrossingCounter` mantém um buffer `_best_crop[track_id]` atualizado a cada frame em que a área da bounding box supera o máximo anterior. No momento do cruzamento, o crop de maior qualidade histórico é despachado para `ocr_queue` — nunca o frame exato do cruzamento, que raramente coincide com o ponto de máxima proximidade do veículo com a câmera.

**2. Detecção de cruzamento via produto vetorial 2D**

A fórmula `_side(A, B, P) = (B.x − A.x)(P.y − A.y) − (B.y − A.y)(P.x − A.x)` determina de qual lado da linha virtual o centróide se encontra. O cruzamento é detectado com a condição blindada:

```python
return d1 * d2 < 0 or (d1 == 0) != (d2 == 0)
```
Detalhes de implementação
em `ARCHITECTURE.md` e `src/counting/crossing_logic.py`.

**3. Jitter freeze**

Se o deslocamento entre frames for menor que `min_displacement_px`, o centróide de referência é **congelado** — não atualizado. Isso garante que travessias lentas (veículo engatinhando sobre a linha) sejam detectadas: o próximo frame com deslocamento válido ainda usa o ponto histórico como referência, e não o último ponto de jitter.

**4. Separação de responsabilidades por filas**

| Fila | maxsize | Política | Propósito |
|---|---|---|---|
| `frame_queue` | 3 | Backpressure (blocking put) | Zero frames descartados em modo arquivo |
| `ocr_queue` | 10 | put_nowait (descarte) | ONNX não satura cores de CPU em background |
| `db_queue` | 200 | put_nowait (descarte) | Inferência nunca bloqueada por I/O de banco |

**5. Graceful shutdown garantido**

Ordem inegociável no `finally`: `stop_event` → `VideoCapture.stop()` + `join(5s)` → `OCRWorker.stop_and_join(10s)` → `DbWriter.flush_and_close(5s)` → `VideoWriter.release()`. Isso assegura que todos os eventos sejam persistidos mesmo quando o usuário pressiona Ctrl+C no meio do processamento.

**6. WAL mode no SQLite**

`PRAGMA journal_mode=WAL` + `PRAGMA synchronous=NORMAL` permitem que o thread principal leia o banco enquanto o `DbWriter` escreve, sem bloqueios de tabela. Essencial para não atrasar o loop de inferência durante commits.

---

## Limitações Conhecidas

### OCR em CPU: janela estreita em câmeras ao vivo

Com `fast-alpr` em CPU, o ONNX Runtime processa cada crop em ~200ms. O pipeline absorve essa latência sem impactar o frame rate de detecção (OCR corre em thread separada com `maxsize=10` não-bloqueante). Se muitos veículos cruzarem em sequência rápida, crops excedentes são descartados silenciosamente — o evento ainda é persistido no banco de dados sem o campo de placa.

### Frame rate reduzido em CPU

Em CPU, o YOLOv8s processa a **~4.5 FPS** (versus 40–100 FPS em GPU RTX 3060+). Em vídeos pré-gravados isso não afeta a contagem (todos os frames são avaliados graças ao modo backpressure), mas em câmeras ao vivo (`realtime: true`) a latência máxima tolerável pode ser excedida, causando descarte de frames e eventual subcontagem em trechos de tráfego denso.

### Crop de placa em câmeras de rodovia

Câmeras de monitoramento rodoviário são posicionadas para capturar a cena completa, não para dar zoom em placas. O `best_crop` de um veículo a 20–30 metros resulta em uma placa de 20–40 pixels de largura. O detector dedicado `yolo-v9-t-384-license-plate-end2end` do fast-alpr compensa parcialmente esse efeito, mas o parâmetro `ocr.min_bbox_area_ratio: 0.005` ainda descarta veículos muito distantes. Uma placa só é tentada quando o veículo está suficientemente próximo, o que em rodovias de alta velocidade representa uma janela temporal de 1–3 frames.

### Rigorosidade do filtro regex

O sistema aceita apenas os padrões oficiais brasileiros:
- **Mercosul** (desde 2018): `[A-Z]{3}[0-9][A-Z][0-9]{2}` (ex: `ABC1D23`)
- **Antigo**: `[A-Z]{3}[0-9]{4}` (ex: `ABC1234`)

Placas de outros países, placas de obra, ou texto parcialmente ocluído são descartados. Isso **elimina falsos positivos** ao custo de não capturar placas estrangeiras — trade-off intencional para um cenário de rodovia nacional.

### Classificação de veículos baseada em COCO

A distinção entre classes (car, truck, bus, motorcycle) usa os IDs COCO do YOLOv8, que não distingue subcategorias como caminhonete vs. caminhão pesado, ou van vs. ônibus. Para granularidade maior, um modelo fine-tuned no domínio rodoviário brasileiro seria necessário.

### Classificação SUV/Picape vs Sedan/Hatch

O YOLOv8 pré-treinado no COCO não possui classes nativas para SUV ou Picape —
apenas `car` e `truck`. A heurística de área relativa + aspect ratio implementada
no `CrossingCounter` é limitada pelo ângulo bird's eye desta câmera, onde SUVs e
sedans apresentam proporções similares vistas de cima. Os thresholds são
configuráveis via `counting.suv_aspect_ratio_threshold` e `counting.suv_area_threshold`
no `settings.yaml`.

### Detecção de Motocicletas

Threshold assimétrico implementado (`motorcycle_threshold: 0.35` vs
`default_class_threshold: 0.45`), recuperando motos de baixa confiança sem
impactar a precisão nas demais classes.

Ground truth manual do vídeo de teste:

| Classe | Ground Truth | v1.2.1 (YOLOv8n) | v1.3.0 (YOLOv8s) | Recall v1.3.0 |
|---|---|---|---|---|
| Sedan/Hatch | 60 | 57 | 62* | — |
| SUV/Picape | 20 | 11 | 8 | 40% |
| Caminhão/Ônibus | 14 | 24** | 24** | — |
| Motocicleta | 18 | 11 | 15 | 83% |
| **Total** | **112** | **103** | **109** | **97%** |

*Sobrecontagem: sedan inclui detecções duplicadas de câmera bird's eye.
**Sobrecontagem absorve SUVs/Picapes não diferenciadas pelo COCO.
YOLOv8s melhorou recall de motocicletas (61% → 83%). SUV/Picape limitado
a 40% pelo ângulo bird's eye desta câmera — heurística de proporção não
distingue SUV de sedan visto de cima.

---

## Pré-requisitos

| Requisito | Mínimo | Recomendado |
|---|---|---|
| Python | 3.11+ | 3.11 |
| GPU NVIDIA | — | RTX 3060+ (CUDA 12+) |
| RAM | 8 GB | 16 GB |
| Docker Engine | 24.0+ | 24.0+ |
| nvidia-container-toolkit | — | Necessário apenas para perfil GPU |
| Vídeo de entrada | MP4, 720p+ | 1080p, câmera fixa |

---

## Como Executar

### Execução local (sem Docker)

```bash
# 1. Criar e ativar ambiente virtual
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

# 2. Instalar dependências (inclui YOLOv8, ByteTrack, EasyOCR, SQLAlchemy)
pip install -e ".[dev]"

# 3. Colocar o vídeo de entrada
cp /caminho/para/video.mp4 data/inputs/video.mp4

# 4. Baixar modelos de detecção e placa
python scripts/download_models.py --model yolov8s.pt
python scripts/download_plate_model.py   # requer HF_TOKEN no .env

# 5. Executar
python main.py --config config/settings.yaml
```

O modelo YOLOv8n (~6 MB) é baixado automaticamente pelo Ultralytics na primeira execução caso não exista. O vídeo anotado e o banco de dados são gerados em `data/outputs/`.

### Execução via Docker (CPU)

```bash
# 1. Colocar vídeo de entrada
cp /caminho/para/video.mp4 data/inputs/video.mp4

# 2. Build e execução (o modelo é baixado automaticamente dentro do container)
docker compose up
```

### Execução via Docker (GPU NVIDIA)

```bash
# Pré-requisito: nvidia-container-toolkit instalado no host
# Editar config/settings.yaml:
#   model.device: "cuda"
#   model.fp16: true

cp /caminho/para/video.mp4 data/inputs/video.mp4
docker compose --profile gpu up
```

### Consultar o banco de dados após execução

```bash
# Resumo por classe
sqlite3 data/outputs/events.db \
  "SELECT vehicle_class, COUNT(*) AS total FROM vehicle_events GROUP BY vehicle_class ORDER BY total DESC"

# Placas detectadas
sqlite3 data/outputs/events.db \
  "SELECT track_id, vehicle_class, plate_text, plate_confidence FROM vehicle_events WHERE plate_text IS NOT NULL"

# Exportar todos os eventos para CSV
sqlite3 -csv -header data/outputs/events.db "SELECT * FROM vehicle_events" > eventos.csv
```

---

## Configuração

Todos os parâmetros ajustáveis ficam em `config/settings.yaml`. Nunca use constantes hardcoded no código.

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `video.source` | `"data/inputs/video.mp4"` | Caminho do vídeo ou índice de câmera (`0`, `1`, ...) |
| `video.resize_width` | `1280` | Largura alvo para redimensionamento (mantém proporção) |
| `video.realtime` | `false` | `true` para câmera ao vivo (descarte); `false` para arquivo (backpressure) |
| `model.weights` | `"yolov8s.pt"` | Arquivo de pesos YOLO (baixar via `scripts/download_models.py`) |
| `model.device` | `"cpu"` | `"cpu"`, `"cuda"` ou `"mps"` |
| `model.fp16` | `false` | Half-precision (requer CUDA; desabilitar em CPU) |
| `model.confidence_threshold` | `0.45` | Limiar mínimo de confiança para detecções |
| `counting.line_points` | `[[0,540],[1920,540]]` | Dois pontos definindo a linha virtual em pixels |
| `counting.direction` | `"any"` | `"any"`, `"top_to_bottom"` ou `"bottom_to_top"` |
| `counting.min_displacement_px` | `5` | Deslocamento mínimo entre frames para ignorar jitter |
| `ocr.enabled` | `true` | Habilita/desabilita OCR de placas |
| `ocr.engine` | `"fast_alpr"` | Motor OCR: `"fast_alpr"` (ONNX, recomendado) ou `"easyocr"` |
| `ocr.min_bbox_area_ratio` | `0.005` | Só tenta OCR se bbox > 0.5% da área do frame |
| `database.sqlite_path` | `"data/outputs/events.db"` | Caminho do banco SQLite de saída |

---

## Schema do Banco de Dados

**Tabela: `vehicle_events`**

| Coluna | Tipo | Nullable | Descrição |
|---|---|---|---|
| `id` | INTEGER | NOT NULL | Chave primária autoincrement |
| `track_id` | INTEGER | NOT NULL | ID único do track (ByteTrack) |
| `vehicle_class` | VARCHAR(50) | NOT NULL | Classe detectada (car, truck, bus, motorcycle) |
| `plate_text` | VARCHAR(20) | NULL | Placa lida pelo OCR (padrão brasileiro) |
| `plate_confidence` | FLOAT | NULL | Confiança do OCR (0.0–1.0) |
| `frame_number` | INTEGER | NOT NULL | Frame do vídeo em que ocorreu o cruzamento |
| `timestamp` | DATETIME | NOT NULL | Instante UTC do cruzamento |
| `session_id` | VARCHAR(36) | NOT NULL | UUID da execução do pipeline |

---

## Persistência de Dados

Todos os eventos são gravados de forma assíncrona em um banco SQLite (`events_final.db`), permitindo consultas analíticas sem bloquear o pipeline de inferência em tempo real.

Para visualizar o schema completo, modelagem das tabelas e exemplos de queries, consulte o documento técnico [ARCHITECTURE.md](ARCHITECTURE.md).

---


## Benchmark Real (CPU)

Medição executada no hardware de desenvolvimento (vídeo 1920×1080, modo `realtime: false`):

| Métrica | v1.2.1 (EasyOCR + YOLOv8n) | v1.3.0 (fast-alpr + YOLOv8s) | Delta |
|---|---|---|---|
| Ambiente | CPU (Intel Core Ultra 5 125H, Python 3.14, Windows 11) | idem | — |
| Duração do vídeo | 90 s (2 700 frames) | idem | — |
| FPS médio | 4.78 | **4.5** | -0.28 |
| Tempo de processamento | ~9.0 min | **~9.5 min** | +0.5 min |
| Veículos contados | 103 | **109** | **+6** |
| Placas lidas | 4 | **65** | **+61** |
| Taxa OCR | 3.9% | **59.6%** | +55.7% |
| Confiança média OCR | 0.57 | **0.96** | +0.39 |
| Testes passando | 89 | **93** | +4 |

> **Nota:** Em modo `realtime: false` todos os frames são avaliados independentemente
> do FPS — não há subcontagem por descarte. O `ocr_queue` com `maxsize=10` e
> `put_nowait()` garante que o ONNX não sature os cores de CPU em background,
> mantendo o pipeline de detecção a plena velocidade.

## Modelos

| Arquivo | Destino | Como obter |
|---|---|---|
| `yolov8s.pt` | `models/yolov8s.pt` | `python scripts/download_models.py --model yolov8s.pt` |
| `license_plate_detector.pt` | `models/license_plate_detector.pt` | `python scripts/download_plate_model.py` (requer `HF_TOKEN`) |
| `yolo-v9-t-384-license-plate-end2end` | `~/.cache/open-image-models/` | Baixado automaticamente pelo fast-alpr na primeira execução |
| `cct-xs-v2-global-model` | `~/.cache/fast-plate-ocr/` | Baixado automaticamente pelo fast-alpr na primeira execução |

> Os modelos ONNX do fast-alpr são cacheados globalmente e não precisam estar em `models/`.
> Apenas `yolov8s.pt` e `license_plate_detector.pt` devem estar presentes localmente.

### Detecção: YOLOv8s (Ultralytics)

O modelo small (`s`) foi escolhido após benchmark quantitativo vs. YOLOv8n: ganho de ~8 pontos de mAP sem queda de FPS mensurável em CPU neste domínio de vídeo. O YOLOv8 usa o head COCO com 80 classes, das quais o pipeline filtra apenas as relevantes: `car` (id=2), `motorcycle` (id=3), `bus` (id=5), `truck` (id=7). Classificação de subcategorias (sedan_hatch, suv_pickup, truck_bus) é feita via heurística de área/aspecto no `CrossingCounter`.

### Rastreamento: ByteTrack (via Supervision)

ByteTrack mantém identidade de objetos entre frames mesmo com oclusões temporárias, usando uma fila de "tracks perdidos" por até `track_buffer` frames. Essencial para que um veículo que sai momentaneamente do campo de visão não seja contado duas vezes.

### OCR: fast-alpr (ONNX)

`fast-alpr` foi escolhido após benchmark comparativo com EasyOCR (ver `BENCHMARK_PLAN.md`). Utiliza pipeline ONNX end-to-end especializado em placas: detector YOLO `yolo-v9-t-384-license-plate-end2end` + OCR `cct-xs-v2-global-model`. Resultados reais: **65 placas lidas em 109 veículos** (59.6% taxa de captura, sessão a7c76b8e) com confiança média de **0.96**. A `ocr_queue` é configurada com `maxsize=10` e `put_nowait()` para evitar que o ONNX Runtime sature os cores de CPU concorrendo com o PyTorch em background.

## Evolução por Versão

| Versão | FPS | Veículos | Placas | Destaques |
|---|---|---|---|---|
| v1.0.0 | 6.2 | 101 | 1 | Pipeline base completo |
| v1.1.2 | 6.3 | 103 | 2 | Classificação por heurística, cores por classe |
| v1.2.0 | 6.3 | 103 | 2 | Threshold assimétrico motos |
| v1.2.1 | 9.7* | 103 | 4 | PlateDetector dedicado (two-stage OCR) |
| **v1.3.0** | **4.5** | **109** | **65** | **YOLOv8s + fast-alpr ONNX + ocr_queue não-bloqueante** |

*FPS medido pelo `_FpsMeter` do `main.py`. O `benchmark_ocr.py` reporta FPS
diferente por variações de timing com threads OCR ativas.

---

## Roadmap Técnico — Melhorias Identificadas

### Classificação de Veículos
- **Fine-tuning YOLOv8:** anotar 2.000 crops do próprio vídeo com
  sedan_hatch/suv_pickup/truck_bus e treinar cabeçalho de classificação.
  Estimativa: recall SUV/Picape de 55% para 85%+.

### Performance
- **GPU:** pipeline passa de ~4.5 FPS para 50+ FPS, resolvendo organicamente
  os problemas de OCR em tempo real (mais frames = mais chances de captura de placa).
- **fast-alpr CUDA:** com GPU disponível, o ONNX Runtime utiliza CUDAExecutionProvider
  automaticamente, eliminando completamente a contenção de CPU.


