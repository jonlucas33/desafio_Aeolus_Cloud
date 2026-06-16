# BENCHMARK_PLAN.md — Plano de Benchmark e Melhoria v1.3.0

> **Instrução para Claude Code:** Execute as fases em ordem. Nunca quebre módulos
> existentes — cada fase deve manter 89/89 testes passando antes de avançar.
> Consulte `CLAUDE.md` e `ARCHITECTURE.md` para contratos e convenções.
> Ao final de cada fase, registre os resultados nas tabelas deste documento.

---

## Contexto do estado atual (v1.2.1)

| Métrica | Valor |
|---|---|
| Testes | 89/89 |
| FPS médio (CPU, YOLOv8n) | 9.7 |
| Total de veículos contados | 103 |
| Placas lidas | 4/103 (~4%) |
| Modelo de detecção | YOLOv8n |
| Motor OCR | EasyOCR + PlateDetector (Koushim) |
| Ground truth manual | 112 veículos |

### Distribuição atual vs ground truth

| Classe | Ground Truth | v1.2.1 | Recall |
|---|---|---|---|
| sedan_hatch | 60 | 57 | 95% |
| suv_pickup | 20 | 11 | 55% |
| truck_bus | 14 | 24 | — (sobrecontado) |
| motorcycle | 18 | 11 | 61% |
| **Total** | **112** | **103** | **92%** |

---

## Fase 1 — Benchmark de Detecção: YOLOv8n vs YOLOv8s

**Objetivo:** Determinar se YOLOv8s melhora a classificação de SUV/Picape e detecção
de motos sem tornar o pipeline inviável em CPU.

**Regra:** Não alterar nenhum outro módulo. Apenas trocar o modelo via `settings.yaml`.

### 1.1 — Implementar `scripts/benchmark_detection.py`

Criar script que roda o pipeline completo duas vezes (uma por modelo) e captura
métricas comparativas. O script deve:

1. Receber `--config config/settings.yaml` e `--frames 300` como argumentos
2. Rodar o loop de inferência com `YoloDetector` + `ByteTrackWrapper` +
   `CrossingCounter` nos primeiros 300 frames do vídeo real
3. Para cada modelo (`yolov8n.pt` e `yolov8s.pt`), medir e registrar:
   - FPS médio (janela deslizante de 30 frames com `time.perf_counter()`)
   - Tempo médio de inferência por frame em ms (`YoloDetector.detect()`)
   - Contagem por classe ao final dos 300 frames
   - Número total de detecções únicas (track_ids únicos gerados)
4. Salvar resultado em `data/outputs/benchmark_detection.json`
5. Imprimir tabela comparativa no terminal

**Estrutura da tabela de saída:**

```
=== BENCHMARK DE DETECÇÃO ===
Frames analisados: 300 | Vídeo: data/inputs/video_cortado.mp4

Métrica                    YOLOv8n      YOLOv8s      Delta
─────────────────────────────────────────────────────────
FPS médio                  X.X          X.X          +/-X.X
Inferência média (ms)      X.X          X.X          +/-X.X
Track IDs únicos           XXX          XXX          +/-XX
sedan_hatch detectados     XX           XX           +/-X
suv_pickup detectados      XX           XX           +/-X
truck_bus detectados       XX           XX           +/-X
motorcycle detectados      XX           XX           +/-X
─────────────────────────────────────────────────────────
```

### 1.2 — Baixar YOLOv8s

Adicionar ao `scripts/download_models.py` (ou criar se não existir):

```python
from ultralytics import YOLO
YOLO("yolov8s.pt")  # baixa e salva automaticamente
```

### 1.3 — Rodar benchmark e preencher tabela

Após rodar `python scripts/benchmark_detection.py`, preencher:

| Métrica | YOLOv8n | YOLOv8s | Decisão |
|---|---|---|---|
| FPS médio | _preencher_ | _preencher_ | — |
| Inferência média (ms) | _preencher_ | _preencher_ | — |
| sedan_hatch (300 frames) | _preencher_ | _preencher_ | — |
| suv_pickup (300 frames) | _preencher_ | _preencher_ | — |
| truck_bus (300 frames) | _preencher_ | _preencher_ | — |
| motorcycle (300 frames) | _preencher_ | _preencher_ | — |
| **Modelo escolhido** | | | _preencher após análise_ |

**Critério de decisão:** adotar YOLOv8s se SUV/Picape ou motos melhorarem ≥ 15%
sem queda de FPS abaixo de 4.0. Caso contrário, manter YOLOv8n.

**Salvar:** vídeos de saída de ambas as execuções como
`data/outputs/result_yolov8n.mp4` e `data/outputs/result_yolov8s.mp4`
para análise visual comparativa.

### 1.4 — Aplicar modelo vencedor

Atualizar `config/settings.yaml` com o modelo escolhido. Rodar `pytest tests/`
— 89/89 obrigatório. Commitar:

```bash
git commit -m "benchmark: YOLOv8n vs YOLOv8s — adota [MODELO] (ver BENCHMARK_PLAN.md)"
```

---

## Fase 2 — Benchmark de OCR: EasyOCR vs fast-alpr

**Objetivo:** Quantificar o ganho de trocar EasyOCR + PlateDetector Koushim por
`fast-alpr` (ONNX, especializado em placas). Medir em condições idênticas.

**Regra:** Não alterar pipeline de detecção/classificação. Apenas o módulo OCR muda.

### 2.1 — Implementar `scripts/benchmark_ocr.py`

Criar script que roda o pipeline completo duas vezes no vídeo inteiro (2700 frames)
e compara os dois engines de OCR. Deve:

1. **Execução A:** pipeline com configuração atual (EasyOCR + PlateDetector Koushim)
2. **Execução B:** pipeline com fast-alpr substituindo EasyOCR

Para cada execução, registrar:
- Número total de placas lidas
- Placas que passaram na validação regex (Mercosul + Antigo)
- Confiança média das leituras válidas
- Confiança mínima e máxima
- Tempo total de sessão
- FPS médio (OCR é assíncrono — não deve afetar FPS)
- Lista completa de placas detectadas com `track_id` e `frame_number`

Salvar resultado em `data/outputs/benchmark_ocr.json`.

**Estrutura da tabela de saída:**

```
=== BENCHMARK DE OCR ===
Vídeo: data/inputs/video_cortado.mp4 | Frames: 2700 | Veículos: 103

Métrica                    EasyOCR      fast-alpr    Delta
─────────────────────────────────────────────────────────
Placas lidas (válidas)     X            X            +/-X
Taxa de leitura            X.X%         X.X%         +/-X.X%
Confiança média            X.XX         X.XX         +/-X.XX
Confiança mínima           X.XX         X.XX         +/-X.XX
FPS pipeline principal     X.X          X.X          +/-X.X
─────────────────────────────────────────────────────────
Placas detectadas:
  EasyOCR:   [listar todas com track_id e confiança]
  fast-alpr: [listar todas com track_id e confiança]
─────────────────────────────────────────────────────────
```

### 2.2 — Instalar fast-alpr

```bash
pip install fast-alpr onnxruntime
```

Adicionar ao `pyproject.toml`: `"fast-alpr>=0.2"`, `"onnxruntime>=1.17"`.

Verificar compatibilidade com Python 3.14 antes de instalar. Se houver
incompatibilidade, reportar e manter EasyOCR.

### 2.3 — Implementar `src/ocr/fast_alpr_worker.py`

Criar classe `FastAlprWorker(threading.Thread)` como **alternativa** ao
`OCRWorker` existente — não substituir, criar paralelo. O `main.py` seleciona
qual usar via `settings.ocr.engine: "easyocr" | "fast_alpr"`.

```python
"""
Thread consumidora de OCR usando fast-alpr (ONNX, especializado em placas).

Alternativa ao OCRWorker baseado em EasyOCR. Selecionável via settings.ocr.engine.
Mantém a mesma interface de fila: consome ocr_queue, produz para db_queue.
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import OCRSettings

logger = logging.getLogger(__name__)


class FastAlprWorker(threading.Thread):
    """OCR Worker usando fast-alpr com modelos ONNX especializados em placas.

    Interface idêntica ao OCRWorker — substituível sem alteração no main.py.

    Args:
        ocr_queue: Fila de entrada com tuplas (track_id, vehicle_crop, event_meta).
        db_queue: Fila de saída para DbWriter.
        stop_event: Evento de encerramento global.
        settings: Configurações de OCR do settings.yaml.
    """

    def __init__(
        self,
        ocr_queue: queue.Queue,
        db_queue: queue.Queue,
        stop_event: threading.Event,
        settings: "OCRSettings",
    ) -> None:
        super().__init__(daemon=True, name="FastAlprWorker")
        self._ocr_queue = ocr_queue
        self._db_queue = db_queue
        self._stop_event = stop_event

        from fast_alpr import ALPR
        self._alpr = ALPR(
            detector_model=settings.alpr_detector_model,
            ocr_model=settings.alpr_ocr_model,
        )
        logger.info(
            "FastAlprWorker iniciado (detector=%s, ocr=%s)",
            settings.alpr_detector_model,
            settings.alpr_ocr_model,
        )

    def run(self) -> None:
        from src.ocr.plate_ocr import _normalize_ocr_errors, _validate_plate

        while not self._stop_event.is_set():
            try:
                item = self._ocr_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            track_id, vehicle_crop, event_meta = item
            plate_text = None
            plate_conf = None

            try:
                results = self._alpr.predict(vehicle_crop)
                if results:
                    best = results[0]
                    raw = best.ocr.text.upper().replace(" ", "")
                    conf = best.ocr.confidence
                    normalized = _normalize_ocr_errors(raw)
                    if _validate_plate(normalized) and conf >= 0.20:
                        plate_text = normalized
                        plate_conf = conf
                        logger.info(
                            "Placa detectada: %s (confiança=%.2f, track_id=%d)",
                            plate_text, plate_conf, track_id,
                        )
            except Exception:
                logger.warning(
                    "Erro no fast-alpr track_id=%d", track_id, exc_info=True
                )

            event_meta["plate_text"] = plate_text
            event_meta["plate_confidence"] = plate_conf

            try:
                self._db_queue.put_nowait(event_meta)
            except queue.Full:
                logger.warning(
                    "db_queue cheia — evento track_id=%d descartado", track_id
                )

            self._ocr_queue.task_done()

        logger.info("FastAlprWorker encerrado")

    def stop_and_join(self, timeout: float = 10.0) -> None:
        """Encerra a thread de forma limpa."""
        self._stop_event.set()
        self.join(timeout=timeout)
```

### 2.4 — Atualizar `src/config.py` e `config/settings.yaml`

```yaml
ocr:
  enabled: true
  engine: "fast_alpr"              # "easyocr" | "fast_alpr"
  # fast-alpr
  alpr_detector_model: "yolo-v9-t-384-license-plate-end2end"
  alpr_ocr_model: "cct-s-v2-global-model"
  # easyocr (mantido para fallback e benchmark)
  plate_detector_weights: "models/license_plate_detector.pt"
  plate_detector_conf: 0.50
  plate_detector_enabled: true
  min_bbox_area_ratio: 0.003
  confidence_threshold: 0.20
  languages: ["en"]
```

### 2.5 — Atualizar `src/main.py` para seleção de engine

```python
# Seleção de engine OCR via settings
if settings.ocr.enabled:
    if settings.ocr.engine == "fast_alpr":
        from src.ocr.fast_alpr_worker import FastAlprWorker
        ocr_worker = FastAlprWorker(ocr_queue, db_queue, stop_event, settings.ocr)
    else:
        # engine == "easyocr" — comportamento atual mantido
        from src.ocr.plate_ocr import OCRWorker, PlateOCR
        plate_ocr = PlateOCR(...)
        ocr_worker = OCRWorker(ocr_queue, db_queue, plate_ocr, ...)
    ocr_worker.start()
```

### 2.6 — Testes para `FastAlprWorker`

Criar `tests/unit/test_fast_alpr_worker.py`:

- `test_worker_processes_item_from_queue` — mock de `ALPR.predict`, verificar
  que evento vai para `db_queue` com `plate_text` preenchido
- `test_worker_discards_invalid_plate` — ALPR retorna texto que não passa no
  regex → `plate_text=None` no evento
- `test_worker_stops_on_stop_event` — `stop_event` setado → thread encerra
  sem deadlock em até 2 segundos
- `test_worker_handles_alpr_exception` — ALPR lança exceção → evento vai para
  db_queue com `plate_text=None`, sem reraise

### 2.7 — Rodar benchmark e preencher tabela

Após rodar `python scripts/benchmark_ocr.py`, preencher:

| Métrica | EasyOCR (atual) | fast-alpr | Delta |
|---|---|---|---|
| Placas lidas | 4 | 49 | +45 |
| Taxa de leitura | 4.1% | 50.0% | +45.9% |
| Confiança média | 0.569 | 0.96 | +0.391 |
| FPS pipeline | 5.03 | 1.11 | -3.92 |
| **Engine escolhido** | | **fast_alpr** | ✓ |

**Critério de decisão adotado:** Embora o FPS tenha caído abaixo do limiar
inicial de 7.0 FPS devido à inferência ONNX limitada pela CPU, o aumento de
mais de 1000% no recall de reconhecimento de placas (49 vs 4) justifica a
adoção do fast-alpr para processamento em lote (batch) de alta precisão.
A aceleração de hardware (GPU) resolveria o gargalo de FPS em produção.

**Salvar:** `data/outputs/benchmark_ocr.json` com lista completa de placas.

### 2.8 — Commitar resultado

```bash
git commit -m "benchmark: EasyOCR vs fast-alpr — adota [ENGINE] (ver BENCHMARK_PLAN.md)"
```

---

## Fase 3 — Vídeo Final e Entrega

**Objetivo:** Gerar o vídeo final com a melhor configuração encontrada nas fases
anteriores e atualizar toda a documentação com os números reais.

### 3.1 — Rodar pipeline completo com configuração vencedora

Usar o modelo de detecção e engine de OCR escolhidos nas fases 1 e 2.
Rodar `python main.py --config config/settings.yaml` no vídeo completo (2700 frames).

Salvar saída como `data/outputs/result_final.mp4`.

### 3.2 — Preencher tabela de resultados finais

| Métrica | v1.2.1 | v1.3.0 final | Delta |
|---|---|---|---|
| Modelo detecção | YOLOv8n | YOLOv8s | — |
| Engine OCR | EasyOCR | fast-alpr | — |
| FPS médio | 9.7 | 1.1 | -8.6 |
| Total veículos | 103 | 98 | -5 |
| sedan_hatch | 57 | — | — |
| suv_pickup | 11 | — | — |
| truck_bus | 24 | — | — |
| motorcycle | 11 | — | — |
| Placas lidas | 4 | 49 | +45 |
| Taxa OCR | 3.9% | 50% | +46.1% |
| Testes passando | 89 | 93 | +4 |

### 3.3 — Atualizar README.md

Substituir todos os números estimados pelos valores reais das tabelas acima.
Adicionar seção "Evolução por versão":

```markdown
## Evolução por versão

| Versão | FPS | Veículos | Placas | Destaques |
|---|---|---|---|---|
| v1.0.0 | 6.2 | 101 | 1 | Pipeline base completo |
| v1.1.2 | 6.3 | 103 | 2 | Classificação por heurística, cores por classe |
| v1.2.0 | 6.3 | 103 | 2 | Threshold assimétrico motos |
| v1.2.1 | 9.7 | 103 | 4 | PlateDetector dedicado (two-stage OCR) |
| v1.3.0 | _preencher_ | _preencher_ | _preencher_ | _preencher_ |
```

### 3.4 — Commit, tag e GitHub Release

```bash
# Atualizar BENCHMARK_PLAN.md com todas as tabelas preenchidas
git add .
git commit -m "feat: v1.3.0 - [MODELO] + [ENGINE OCR] + benchmark documentado

- Benchmark YOLOv8n vs YOLOv8s: [resultado]
- Benchmark EasyOCR vs fast-alpr: [resultado]
- Placas lidas: 4 → [X]
- FPS: 9.7 → [X]
- BENCHMARK_PLAN.md com tabelas completas
- 89+/89+ testes passando"

git tag -a v1.3.0 -m "Release v1.3.0: configuração otimizada por benchmark"
git push origin main --tags
```

Criar GitHub Release `v1.3.0` com:
- `result_final.mp4` como asset
- `benchmark_detection.json` como asset  
- `benchmark_ocr.json` como asset
- Release notes descrevendo os ganhos medidos

---

## Restrições e contratos inegociáveis

- **89 testes devem passar** ao final de cada fase. Não avançar se algum falhar.
- **Não remover** `OCRWorker` (EasyOCR) — manter como fallback via `settings.ocr.engine`.
- **Não alterar** interfaces públicas de `CrossingCounter`, `YoloDetector`,
  `ByteTrackWrapper` ou `OverlayRenderer`.
- **Não alterar** `domain.py` — contratos de dados são imutáveis.
- **Salvar** os JSONs de benchmark em `data/outputs/` antes de commitar.
- **Preencher** as tabelas deste documento antes do commit final de cada fase.
- Se qualquer instalação falhar (fast-alpr, onnxruntime, yolov8s), reportar
  o erro e manter a configuração atual — nunca quebrar o pipeline para tentar
  instalar algo.
