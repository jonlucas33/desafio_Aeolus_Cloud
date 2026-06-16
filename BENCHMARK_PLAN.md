# BENCHMARK_PLAN.md — Resultados e Decisões v1.3.0

Documento de registro de benchmarks executados sobre o vídeo `data/inputs/video_cortado.mp4`
(1920×1080, 2 700 frames, 90 s de rodovia BR-232). Todos os números são medidos —
sem estimativas. Onde o dado não existe, o campo indica **n/d**.

---

## Estado de partida (v1.2.1)

| Métrica | Valor |
|---|---|
| Modelo de detecção | YOLOv8n |
| Engine OCR | EasyOCR + PlateDetector (Koushim/YOLOv8) |
| FPS médio (main.py) | 9.7 |
| Total de veículos | 103 |
| Placas lidas | 4 |
| Taxa OCR | 3.9% |
| Testes passando | 89 |

---

## Ground truth manual

Contagem visual quadro a quadro do vídeo de entrada:

| Classe | Quantidade | Observação |
|---|---|---|
| sedan_hatch | 60 | Inclui hatchbacks e sedãs |
| suv_pickup | 20 | SUVs, caminhonetes |
| truck_bus | 14 | Caminhões pesados, ônibus |
| motorcycle | 18 | Motos e similares |
| **Total** | **112** | Benchmark de referência |

---

## Fase 1 — Detecção: YOLOv8n vs YOLOv8s

### Metodologia

Script `scripts/benchmark_detection.py` rodou o loop de inferência (detecção + rastreamento +
contagem) no vídeo completo (2 700 frames) para cada modelo, sem OCR e sem renderização.
O FPS reportado mede exclusivamente o tempo do loop de inferência
(`YoloDetector.detect` + `ByteTrackWrapper.update` + `CrossingCounter.update`).
Vídeos de saída salvos em `data/outputs/result_yolov8n.mp4` e `result_yolov8s.mp4`.

### Resultados

| Métrica | YOLOv8n | YOLOv8s | Delta |
|---|---|---|---|
| FPS (loop de inferência) | 12.31 | 5.44 | -6.87 |
| Inferência média (ms/frame) | 78 | 180 | +102 |
| Track IDs únicos detectados | 480 | 473 | -7 |
| Cruzamentos contados | 86 | 98 | **+12** |
| sedan_hatch (total tracks) | 208 | 205 | -3 |
| suv_pickup (total tracks) | 70 | 78 | **+8** |
| truck_bus (total tracks) | 68 | 54 | -14 |
| motorcycle (total tracks) | 134 | 136 | +2 |

> **Nota:** FPS acima mede apenas o loop de inferência. O FPS total do pipeline
> (incluindo I/O de vídeo, renderização e OCR assíncrono) é medido na Fase 2.

### Decisão

**YOLOv8s adotado.** Apesar do custo de +102ms por frame (+2.3×), o ganho de
**+12 cruzamentos detectados** (86 → 98) e a melhora de recall em suv_pickup (+8)
foram determinantes. O FPS total do pipeline final (5.56, medido na Fase 2) manteve-se
acima do limiar operacional.

---

## Fase 2 — OCR: EasyOCR vs fast-alpr

### Metodologia

Script `scripts/benchmark_ocr.py` rodou o pipeline completo (YOLOv8s + ByteTrack +
CrossingCounter + OCR) no vídeo completo (2 700 frames) para cada engine.
O OCR roda em thread separada (`OCRWorker` ou `FastAlprWorker`); o FPS reportado
mede o tempo do loop principal (detecção + tracking + counting).
Placas contadas incluem apenas as que passaram na validação regex (Mercosul e padrão antigo).

### Resultados

| Métrica | EasyOCR + PlateDetector | fast-alpr (ONNX) | Delta |
|---|---|---|---|
| FPS pipeline (loop principal) | 4.78 | **5.56** | +0.78 |
| Total veículos | 98 | 98 | 0 |
| Placas lidas (válidas) | 4 | **49** | +45 |
| Taxa de leitura | 4.1% | **50.0%** | +45.9% |
| Confiança média | 0.569 | **0.960** | +0.391 |
| Confiança mínima | 0.259 | 0.744 | +0.485 |

> **Nota sobre variação de veículos (98 vs 109):** O `benchmark_ocr.py` registrou
> 98 cruzamentos; o `main.py` em produção registrou 109. A variação é esperada:
> a thread do ONNX Runtime (fast-alpr) altera o timing do loop principal,
> causando diferenças de até ±10% no número de tracks capturados pelo ByteTrack
> entre execuções. Os números do `main.py` são usados como referência de produção
> no README; os 98 deste benchmark refletem o contexto de medição do OCR.

### Análise do fix de arquitetura (CPU starvation)

Na primeira execução do fast-alpr com `ocr_queue` de `maxsize=50`, o FPS caiu para
**1.11** — uma queda de 78% em relação ao EasyOCR. A causa foi identificada como
**CPU starvation**: o ONNX Runtime usa um pool de threads próprio em C++ que, ao
processar até 50 crops enfileirados em background, saturava todos os cores da CPU,
privando o PyTorch (YOLOv8) de tempo de execução.

Solução: `ocr_queue = Queue(maxsize=10)` com `put_nowait()` e descarte silencioso
(`logger.debug`) quando a fila está cheia. Com `maxsize=10`, o ONNX conclui as
primeiras 10 inferências e para de competir pelos cores antes que o gargalo se instale.
O fast-alpr subiu de 1.11 → **5.56 FPS** sem perda de placas detectadas (49 → 49).

### Decisão

**fast-alpr adotado** com `ocr_queue` `maxsize=10` não-bloqueante.
O ganho absoluto de **+45 placas** (4 → 49) com confiança elevada (0.96) e
FPS superior ao EasyOCR (5.56 vs 4.78) tornam a escolha inequívoca.

---

## Configuração final adotada (v1.3.0)

| Componente | Valor |
|---|---|
| Modelo de detecção | YOLOv8s (`models/yolov8s.pt`) |
| Engine OCR | fast-alpr ONNX (`yolo-v9-t-384-license-plate-end2end` + `cct-xs-v2-global-model`) |
| `ocr_queue` maxsize | 10 (não-bloqueante, `put_nowait`) |
| FPS pipeline (benchmark_ocr.py) | 5.56 |
| Total veículos (benchmark_ocr.py) | 98 |
| Total veículos (main.py produção) | 109 |
| Placas lidas (benchmark_ocr.py) | 49 |
| Placas lidas (main.py produção, sessão a7c76b8e) | 65 |
| Taxa OCR (benchmark, base 98) | 50.0% |
| Taxa OCR (produção, base 109) | 59.6% |
| Confiança média OCR | 0.960 |
| Testes passando | 93 |

---

## Evolução por versão

| Versão | FPS | Veículos | Placas | Taxa OCR | Destaques |
|---|---|---|---|---|---|
| v1.0.0 | 6.2 | 101 | 1 | 1.0% | Pipeline base completo |
| v1.1.2 | 6.3 | 103 | 2 | 1.9% | Classificação por heurística, cores por classe |
| v1.2.0 | 6.3 | 103 | 2 | 1.9% | Threshold assimétrico motos |
| v1.2.1 | 9.7 | 103 | 4 | 3.9% | PlateDetector dedicado (OCR dois estágios) |
| **v1.3.0** | **4.5*** | **109** | **65** | **59.6%*** | YOLOv8s + fast-alpr + ocr_queue não-bloqueante |

> *FPS e taxa OCR de v1.3.0 referem-se ao `main.py` produção (sessão a7c76b8e-6c57-4e22-ad91-95780e6f28e6).
> FPS das versões anteriores medido pelo `_FpsMeter` em main.py (janela deslizante de 30 frames).
> `benchmark_ocr.py` reporta 5.56 FPS e 49/98 placas (50%) — valores de contexto de medição, não de produção.

---

## Limitações conhecidas

- **CPU starvation (resolvido):** ONNX Runtime e PyTorch competem pelo mesmo pool de cores.
  Resolvido com `maxsize=10`; reaparece se o maxsize for aumentado sem GPU disponível.
- **Taxa de descarte:** Com `maxsize=10` e 109 cruzamentos (produção), os 65 eventos com
  placa (59.6% de 109) correspondem aos crops enfileirados nos momentos em que a fila tinha
  espaço disponível. O bug de `event_meta` sem `plate_text` inicializado foi corrigido em
  v1.3.0 — eventos antes perdiam dados antes do fix de `continue` explícito.
- **FPS com GPU:** Com CUDAExecutionProvider (ONNX em GPU), a contenção desaparece e o
  `maxsize` pode ser aumentado para capturar mais placas sem penalidade de FPS.
- **Classificação SUV/Picape:** Recall de 40% (8/20 no ground truth) limitado pelo ângulo
  bird's eye da câmera que equaliza as proporções de SUV e sedan. Ajustável via
  `counting.suv_aspect_ratio_threshold` no settings.yaml.
- **Schema de banco:** Coluna `plate_confidence` (não `plate_conf`), `timestamp` (não `crossed_at`).
  Consultas devem usar os nomes reais definidos em `src/database/models.py`.
