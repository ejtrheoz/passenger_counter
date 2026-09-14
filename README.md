# passenger_counter

FastAPI-сервис: GPU-детекция для одного уже нарезанного клипа активности (door-flow
вход/выход людей — BPJDet + ByteTrack — и предъявление ксивы — YOLO). Сервис не занимается
нарезкой видео на сегменты и не хранит job'ы — это зона ответственности `passenger_aggregator`
(веб-бэкенд с фронтендом), который делит видео на сегменты и по очереди/параллельно шлёт их
сюда по одному клипу за запрос — то, что удобно для serverless GPU-воркеров с лимитами на
размер запроса и время выполнения.

## Эндпоинты

Полный список входов/выходов — в `openapi.json` (сгенерирован из `app/main.py`) и в интерактивном
Swagger UI на `/docs` запущенного сервиса.

| Метод | Путь | Тело (multipart/form-data) | Ответ | Что делает |
|---|---|---|---|---|
| GET | `/health` | — | наличие скриптов/весов, устройство, GPU, список полигонов | |
| POST | `/analyze/people` | `video` (уже нарезанный клип одного интервала активности); полигон двери — `door_polygon` (имя JSON из `assets/door_polygons`, по умолчанию `2.json`) **или** `door_polygon_json` (инлайн `{"points": [[x,y] x4+]}`, например нарисованный пользователем на фронтенде; имеет приоритет над `door_polygon`) | `entered`, `exited`, `door_polygon`, `processing_seconds` | Один ограниченный по времени GPU-вызов на один клип. |
| POST | `/analyze/ksiva` | `video` (уже нарезанный клип) | `unique_ksiva` и статистика детекций | Один ограниченный по времени GPU-вызов на один клип. |

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/analyze/people -F video=@samples/segment_001.mp4 -F door_polygon=2.json
curl -X POST http://localhost:8000/analyze/people -F video=@samples/segment_001.mp4 \
  -F door_polygon_json='{"points": [[10,10],[300,10],[300,200],[10,200]]}'
curl -X POST http://localhost:8000/analyze/ksiva  -F video=@samples/segment_001.mp4
```

Видео обрабатываются по одному (GPU 4 ГБ); запрос блокируется до конца обработки клипа.



## Ассеты

Каталог `assets/` (веса и полигоны, ~75 МБ, закоммичены в репозиторий и запекаются в
образ через `COPY assets ./assets`):

```
assets/
  door_polygons/2.json, 27.json
  models/bpjdet/ch_face_s_1536_e150_best_mMR.pt   # ~25 МБ
  models/ksiva/best.pt                            # ~50 МБ, runs/ksiva_yolo11_final/train/weights/best.pt
```

Исходники детектора людей BPJDet (`third_party/BPJDet`, нужны, потому что `.pt` — pickle
классов модели) в образ клонируются при сборке с github.com/hnuzhy/BPJDet; для запуска без
Docker склонируй их сам в `third_party/BPJDet`.

Пути переопределяются переменными `ASSETS_DIR`, `BPJDET_REPO`, `BPJDET_WEIGHTS`,
`KSIVA_WEIGHTS`, `DOOR_POLYGON_DIR`, `DEFAULT_DOOR_POLYGON`.

## Docker

```bash
docker build -t passenger-counter .

# GPU (нужны NVIDIA-драйвер и NVIDIA Container Toolkit:
#   sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker)
docker run -d --name passenger-counter --gpus all -p 8000:80 passenger-counter

# без GPU
docker run -d --name passenger-counter -p 8000:80 -e DEVICE=cpu passenger-counter

docker logs -f passenger-counter
docker rm -f passenger-counter
```

Веса и код запечены в образ — пересобирать нужно при изменении `app/`, `door-flow/`,
`ksiva/`, `requirements.txt` или `assets/`.

Переменные окружения: `DEVICE` (`0`, `0,1`, `cpu` — при анализе одного клипа используется первое устройство из списка),
`BATCH_SIZE` (4), `WORK_DIR`, `KEEP_ARTIFACTS=1` (не удалять клип/JSON после запроса), `JOB_TIMEOUT_SECONDS` (3600),
`MAX_UPLOAD_MB` (2048).

## Локально без Docker

Нужен Python 3.10 с torch (CUDA), см. `requirements.txt`.

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu130 torch==2.11.0+cu130 torchvision==0.26.0+cu130
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Структура

- `app/` — FastAPI (`main.py` — роуты `/analyze/people` и `/analyze/ksiva`, `pipelines.py` — запуск скриптов,
  `settings.py` — конфиг из env).
- `door-flow/` — `door_flow_counter.py` (детекция+трекинг для одного клипа, GPU) и `aggregate_door_flow.py`
  (автономный CLI-скрипт: активность → клипы → воркер; не вызывается из API, оставлен для локального офлайн-прогона).
- `ksiva/` — `detect_ksiva_video.py` (YOLO для одного клипа, GPU) и `analyze_segment.py` (YOLO + пост-обработка для
  одного уже нарезанного клипа; используется `/analyze/ksiva`). `aggregate_ksiva.py` хранит переиспользуемую
  пост-обработку (`postprocess_detections`, `DETECTION_COLUMNS`) и как автономный CLI-скрипт из API не вызывается.
- `architecture/` — описание конвейеров.
- `openapi.json` — сгенерированная OpenAPI-схема API (актуальный Swagger — на `/docs` запущенного сервиса).

Нарезка видео на сегменты активности, оркестрация job'ов и суммирование результатов по сегментам — в
`passenger_aggregator` (см. его README).
