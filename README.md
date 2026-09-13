# passenger_counter

FastAPI-сервис: загружаешь видео с камеры над дверью автобуса, получаешь количество
вошедших/вышедших людей (door-flow, BPJDet + ByteTrack) и количество уникальных
предъявлений ксивы (YOLO). Сами конвейеры — скрипты в `door-flow/` и `ksiva/`,
API запускает их как подпроцессы и парсит `aggregate.history.json`.

## Эндпоинты

| Метод | Путь | Тело (multipart/form-data) | Ответ |
|---|---|---|---|
| GET | `/health` | — | наличие скриптов/весов, устройство, GPU, список полигонов |
| POST | `/count/people` | `video` (файл), `door_polygon` (опц., имя JSON из `assets/door_polygons`, по умолчанию `2.json`) | `entered`, `exited`, интервалы активности, сегменты |
| POST | `/count/ksiva` | `video` | `unique_ksiva`, статистика детекций |
| POST | `/count/all` | `video`, `door_polygon` (опц.) | `{ "people": …, "ksiva": … }` |

Полигоны: `2.json` — камера layout 1 (видео 1–13), `27.json` — layout 27 (видео 27–48).

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/count/people -F video=@samples/first_1.mp4 -F door_polygon=2.json
curl -X POST http://localhost:8000/count/ksiva  -F video=@samples/first_1.mp4
curl -X POST http://localhost:8000/count/all    -F video=@samples/first_1.mp4
```

Видео обрабатываются по одному (GPU 4 ГБ); запрос блокируется до конца обработки.

## Ассеты

Каталог `assets/` (монтируется в контейнер read-only, веса не в git):

```
assets/
  door_polygons/2.json, 27.json
  models/bpjdet/ch_face_s_1536_e150_best_mMR.pt
  models/ksiva/best.pt                # runs/ksiva_yolo11_final/train/weights/best.pt
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
docker run -d --name passenger-counter --gpus all -p 8000:8000 \
  -v "$PWD/assets:/app/assets:ro" passenger-counter

# без GPU
docker run -d --name passenger-counter -p 8000:8000 -e DEVICE=cpu \
  -v "$PWD/assets:/app/assets:ro" passenger-counter

docker logs -f passenger-counter
docker rm -f passenger-counter
```

`assets/` монтируется как volume — веса/полигоны меняются без пересборки; пересобирать
образ нужно только при изменении `app/`, `door-flow/`, `ksiva/`, `requirements.txt`.

Переменные окружения: `DEVICE` (`0`, `0,1`, `cpu`), `BATCH_SIZE` (4), `WORK_DIR`,
`KEEP_ARTIFACTS=1` (не удалять клипы/JSON после запроса), `JOB_TIMEOUT_SECONDS` (3600),
`MAX_UPLOAD_MB` (2048).

## Локально без Docker

Нужен Python 3.10 с torch (CUDA), см. `requirements.txt`.

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu130 torch==2.11.0+cu130 torchvision==0.26.0+cu130
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Структура

- `app/` — FastAPI (`main.py` — роуты, `pipelines.py` — запуск скриптов, `settings.py` — конфиг из env).
- `door-flow/` — `aggregate_door_flow.py` (активность → клипы → воркер) и `door_flow_counter.py`.
- `ksiva/` — `aggregate_ksiva.py` и `detect_ksiva_video.py`.
- `architecture/` — описание конвейеров.
