# TASKS.md — Backlog de Implementação

> **Instrução para Claude Code:** Execute as tarefas em ordem. Marque `[x]` ao concluir.
> Não avance para a próxima fase sem todos os itens da fase atual marcados e testes passando.
> Em caso de ambiguidade, pergunte antes de implementar.

---

## Fase 1 — Fundação do Projeto

**Objetivo:** Ter o pipeline básico funcionando: capturar frames, detectar veículos, rastrear.
**Critério de conclusão:** Vídeo anotado com bboxes e track_ids sendo gerado sem erros.

- [x] **1.1 — Setup inicial do projeto**
  - Criar `pyproject.toml` com dependências:
    `ultralytics>=8.0`, `paddleocr>=2.7`, `opencv-python-headless>=4.8`,
    `sqlalchemy>=2.0`, `pydantic>=2.0`, `pyyaml`, `numpy`, `pytest`, `pytest-mock`
  - Criar `.gitignore` (excluir: `models/*.pt`, `data/inputs/`, `data/outputs/`, `*.db`, `.env`)
  - Criar `.env.example` com `DATABASE_URL=` e `DB_BACKEND=sqlite`
  - Criar toda a estrutura de diretórios conforme `ARCHITECTURE.md § 2`
  - Criar arquivos `__init__.py` em todos os pacotes `src/`

- [x] **1.2 — Configuração (config/settings.yaml + src/config.py)**
  - Criar `config/settings.yaml` com todos os campos definidos em `ARCHITECTURE.md § CLAUDE.md`
  - Criar `src/config.py` com classes Pydantic `Settings`, `VideoSettings`, `ModelSettings`,
    `TrackingSettings`, `CountingSettings`, `OCRSettings`, `DatabaseSettings`
  - Função `load_settings(path: Path) -> Settings` que carrega e valida o YAML
  - Teste: `tests/unit/test_config.py` — valida que settings.yaml carrega sem erros

- [x] **1.3 — Dataclasses canônicas (src/domain.py)**
  - Implementar `Detection`, `Track`, `VehicleEvent` conforme especificado em `CLAUDE.md § Dataclasses`
  - `Track` deve calcular `centroid` automaticamente via `__post_init__`
  - Nenhuma dependência além de `numpy` e `dataclasses`

- [x] **1.4 — VideoCapture thread (src/capture/video_capture.py)**
  - Implementar classe `VideoCapture` conforme `ARCHITECTURE.md § 4.1`
  - Thread daemon com `threading.Event` para stop limpo
  - Frame drop quando `frame_queue` está cheia (descartar mais antigo)
  - Propriedade `fps` que retorna FPS real da fonte de vídeo
  - Teste: `tests/unit/test_video_capture.py` — mock `cv2.VideoCapture`, verificar que
    frames chegam na fila e que o stop funciona sem deadlock

- [ ] **1.5 — YOLO Detector (src/detection/yolo_detector.py)**
  - Implementar `YoloDetector` conforme `ARCHITECTURE.md § 4.2`
  - Filtrar apenas classes de veículos (car=2, motorcycle=3, bus=5, truck=7)
  - Converter resultado YOLO para `List[Detection]`
  - Método `warmup(frame_shape: tuple, n_frames=3)`: cria frames sintéticos com o shape
    exato do vídeo de entrada e passa pelo mesmo pré-processamento (letterbox) do loop principal.
    Shape diferente causa realocação de memória GPU no primeiro frame real.
  - Teste: `tests/unit/test_yolo_detector.py` — mock do modelo, verificar filtragem de classes

- [ ] **1.6 — ByteTrack Wrapper (src/tracking/bytetrack_wrapper.py)**
  - Implementar `ByteTrackWrapper` conforme `ARCHITECTURE.md § 4.3`
  - Usar `model.track()` com `tracker="bytetrack.yaml"` via Ultralytics, ou
    integrar `supervision.ByteTrack` diretamente
  - Calcular `centroid` no `Track` retornado
  - Teste: `tests/unit/test_bytetrack_wrapper.py` — mock de detecções sequenciais,
    verificar que track_id é mantido entre frames

- [ ] **1.7 — Teste de integração Fase 1**
  - `tests/integration/test_pipeline_phase1.py`
  - Usar vídeo sintético de 30 frames (gerado com `np.zeros`) ou amostra curta do vídeo real
  - Verificar: frames sendo consumidos, detecções retornadas, tracks com IDs consistentes
  - Logar FPS médio no terminal

---

## Fase 2 — Lógica de Negócio e Visualização

**Objetivo:** Pipeline completo com contagem correta e vídeo anotado gerado.
**Critério de conclusão:** Vídeo de saída com linha virtual, bboxes, contador e FPS visíveis.

- [ ] **2.1 — Crossing Logic (src/counting/crossing_logic.py)**
  - Implementar `CrossingCounter` completo conforme `ARCHITECTURE.md § 5`
  - Produto vetorial 2D para detecção de crossing — **não usar threshold em Y**
  - Filtro de jitter (`min_displacement_px`)
  - Filtro de direção (`direction: any | top_to_bottom | bottom_to_top`)
  - Votação majoritária por `track_id` (últimos `class_vote_window` frames)
  - Best-crop buffer: `_best_crop: dict[int, np.ndarray]` e `_max_bbox_area: dict[int, float]`
    atualizados a cada frame. No cruzamento, o `_best_crop` é o payload enviado para o OCR.
  - Testes obrigatórios em `tests/unit/test_crossing_logic.py`:
    - [ ] Veículo cruzando de cima para baixo → contado
    - [ ] Mesmo track_id cruzando duas vezes → contado apenas uma vez
    - [ ] Veículo parado sobre a linha (jitter < `min_displacement_px`) → não contado
    - [ ] Linha diagonal (não horizontal) → crossing detectado corretamente
    - [ ] Cruzamento no sentido errado com `direction=top_to_bottom` → não contado

- [ ] **2.2 — Overlay Renderer (src/rendering/overlay_renderer.py)**
  - Implementar `OverlayRenderer` conforme `ARCHITECTURE.md § 4.7`
  - Desenhar: linha virtual (cor configurável), bbox com cor por classe, track_id acima da bbox,
    nome da classe, contador total (destaque no canto superior esquerdo), FPS (canto superior direito)
  - Paleta de cores por classe: `car=verde`, `suv_pickup=azul`, `truck=vermelho`, `bus=laranja`,
    `motorcycle=amarelo`
  - Sempre trabalhar em `frame.copy()` — nunca modificar o frame original

- [ ] **2.3 — Pipeline principal (src/main.py)**
  - Montar e iniciar todas as threads: `VideoCapture`, inference loop, shutdown limpo
  - Inicializar: `Settings`, `YoloDetector.warmup()`, `ByteTrackWrapper`, `CrossingCounter`,
    `OverlayRenderer`, `cv2.VideoWriter`
  - Loop principal com medição de FPS real (média movel de 30 frames)
  - Graceful shutdown: `Ctrl+C` → parar threads → fechar VideoWriter → logar estatísticas finais
  - Estatísticas finais no terminal: total de veículos por classe, FPS médio, duração do vídeo

- [ ] **2.4 — Teste end-to-end Fase 2**
  - Rodar pipeline no vídeo real de entrada
  - Verificar visualmente que contador sobe corretamente
  - Verificar que vídeo de saída é gerado em `data/outputs/`
  - Medir e logar FPS médio (meta: ≥ 25 FPS em GPU, ≥ 10 FPS em CPU com YOLOv8n)

---

## Fase 3 — Diferenciais (OCR + Banco de Dados)

**Objetivo:** Placas lidas e todos os eventos persistidos no banco.
**Critério de conclusão:** Banco SQLite populado com eventos; placas visíveis no vídeo.

- [ ] **3.1 — Plate OCR (src/ocr/plate_ocr.py)**
  - Implementar `PlateOCR` conforme `ARCHITECTURE.md § 4.5`
  - Pré-processamento: upscale → CLAHE → binarização adaptativa (Otsu)
  - OCR com PaddleOCR (`use_angle_cls=True`, `lang='en'`)
  - Validação com regex: Mercosul `[A-Z]{3}[0-9][A-Z][0-9]{2}` e antigo `[A-Z]{3}[0-9]{4}`
  - Retornar `None` se nenhum padrão válido for encontrado
  - Teste: `tests/unit/test_plate_ocr.py` — crop sintético com texto de placa,
    verificar extração e validação regex

- [ ] **3.2 — Database Models (src/database/models.py)**
  - Implementar `VehicleEvent` SQLAlchemy conforme `ARCHITECTURE.md § 6`
  - Função `init_db(engine)` que cria tabelas e índices
  - Função `create_sqlite_engine(path)` com `connect_args={"check_same_thread": False}`
    e listener de `PRAGMA journal_mode=WAL` + `PRAGMA synchronous=NORMAL` — obrigatório
  - Suporte a PostgreSQL via `DB_BACKEND=postgresql` (engine padrão, sem `check_same_thread`)

- [ ] **3.3 — DB Writer thread (src/database/db_writer.py)**
  - Implementar `DbWriter` como thread Consumer da `db_queue`
  - Batch insert de 10 eventos por vez com `session.bulk_save_objects()`
  - Flush forçado a cada 5 segundos (garante persistência mesmo com poucos veículos)
  - Log de erro (não exceção) se insert falhar — nunca travar o pipeline
  - Método `flush_and_close()` para shutdown limpo

- [ ] **3.4 — Integração OCR + DB no main.py**
  - Inicializar `ThreadPoolExecutor(max_workers=2)` para OCR
  - Inicializar `DbWriter` e sua thread
  - No cruzamento detectado:
    - `CrossingCounter` já mantém `_best_crop[track_id]` atualizado a cada frame
    - Enviar `ocr_queue.put_nowait((track_id, best_crop))` — usa o melhor frame histórico, não o frame atual
    - `db_queue.put_nowait(VehicleEvent(..., plate_text=None))` imediatamente (não bloqueia)
    - Quando OCR retornar resultado: UPDATE no registro existente via `track_id + session_id`
  - Shutdown: `ThreadPoolExecutor.shutdown(wait=True)` → `DbWriter.flush_and_close()`

- [ ] **3.5 — Teste de persistência**
  - Rodar pipeline completo no vídeo real
  - Verificar banco: `SELECT vehicle_class, COUNT(*) FROM vehicle_events GROUP BY vehicle_class`
  - Verificar que placas detectadas têm formato válido
  - Verificar que `session_id` é consistente em toda a execução

---

## Fase 4 — Produção e Entrega

**Objetivo:** Projeto dockerizado, documentado e pronto para avaliação.

- [ ] **4.1 — Dockerfile**
  - Base: `ultralytics/ultralytics:latest`
  - Instalar dependências com `opencv-python-headless` (não `opencv-python`)
  - `COPY` apenas arquivos necessários (excluir `data/`, `models/` via `.dockerignore`)
  - `ENTRYPOINT ["docker/entrypoint.sh"]`
  - `entrypoint.sh`: verificar se `models/yolov8s.pt` existe, baixar se não, executar `main.py`

- [ ] **4.2 — docker-compose.yml**
  - Serviço `vehicle-counter` com volumes: `./data`, `./models`, `./config`
  - Profile `gpu` com `nvidia` device reservation
  - Profile `cpu` com imagem base `python:3.11-slim` e PyTorch CPU
  - Variáveis de ambiente: `DB_BACKEND`, `DATABASE_URL`

- [ ] **4.3 — scripts/download_models.py**
  - Baixar `yolov8s.pt` via `ultralytics.utils.downloads`
  - Verificar SHA256 após download
  - Logar mensagem clara se já existir

- [ ] **4.4 — scripts/benchmark.py**
  - Medir FPS em 4 configurações: GPU+OCR, GPU sem OCR, CPU+OCR, CPU sem OCR
  - Usar os primeiros 300 frames do vídeo de entrada
  - Imprimir tabela de resultados

- [ ] **4.5 — README.md**
  Conteúdo obrigatório:
  - [ ] Descrição do projeto e arquitetura em português
  - [ ] Pré-requisitos (GPU NVIDIA recomendada, Docker, Python 3.11+)
  - [ ] Quickstart com Docker (3 comandos para rodar)
  - [ ] Quickstart sem Docker (virtualenv + pip + python main.py)
  - [ ] Descrição dos parâmetros principais do `settings.yaml`
  - [ ] Tabela de FPS por configuração (preencher com resultado do benchmark)
  - [ ] Descrição dos modelos escolhidos e justificativa
  - [ ] Schema do banco de dados
  - [ ] Exemplos de output (screenshot do vídeo anotado)

- [ ] **4.6 — Revisão final**
  - `pytest tests/` — todos passando
  - `docker compose up` — pipeline roda sem erros
  - Vídeo de saída gerado e visualmente correto
  - Banco populado com dados coerentes
  - README completo e revisado

---

## Notas de Implementação

### Ajuste da linha virtual

A posição da linha virtual (`counting.line_points` no `settings.yaml`) deve ser ajustada
para o vídeo específico recebido. Recomenda-se rodar o pipeline uma vez com `cv2.imshow`
ativo para visualizar e ajustar a posição antes da gravação final.

### Calibração da heurística de classe

O limiar `bbox_area / frame_area > 0.035` para SUV/Picape foi definido como ponto de partida.
Calibrar assistindo o vídeo com o overlay ativo e ajustando o threshold no `settings.yaml`.

### Performance esperada

| Configuração | FPS esperado |
|---|---|
| GPU (RTX 3060+) + YOLOv8s | 40–60 FPS |
| GPU (RTX 3060+) + YOLOv8n | 70–100 FPS |
| CPU (8 cores) + YOLOv8n | 8–15 FPS |
| CPU + YOLOv8s | 3–6 FPS |

OCR em thread separada não impacta FPS do pipeline principal.
