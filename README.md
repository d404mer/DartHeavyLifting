# DartHeavyLifting — отслеживание штанги и позы.

Приложение на Python для детекции и трекинга штанги (вид сбоку) и суставов ног (через MediaPipe Pose) с визуализацией в реальном времени и стримингом данных по UDP (для Unreal Engine и др.).

## Основные возможности

- ✅ Отслеживание позы через MediaPipe Pose (33 ключевые точки) — опционально
- ✅ Расчет углов коленей и отправка всех landmarks
- ✅ Обнаружение штанги по окружности (HoughCircles) + fallback по контурам
- ✅ Оптимизированный трекер штанги: поиск в ROI между запястьями, выбор «лучшего» круга, анти‑дрожание без задержки
- ✅ Визуализация: скелет (если включено), текущая точка штанги и её путь
- ✅ UDP‑стрим в JSON: позиции, углы, путь, уверенность и источник детекции
- ✅ Авто‑очистка пути при простое (если штанга не двигается N секунд)

## Установка

### Стандартная установка
```bash
pip install -r requirements.txt
```

### Проблемы с установкой MediaPipe?

Если возникают ошибки при установке MediaPipe, скорее всего проблема в версии Python.

#### Быстрое решение для Windows:

1. **Проверьте версию Python:**
   ```bash
   python check_python.py
   ```

2. **Если Python 3.12+, установите Python 3.10:**
   
   **Вариант A: Автоматическая установка (рекомендуется)**
   ```bash
   # Запустите скрипт установки Python 3.10
   setup_python310.bat
   
   # Затем настройте проект
   quick_setup.bat
   ```
   
   **Вариант B: Ручная установка через winget**
   ```bash
   winget install Python.Python.3.10
   py -3.10 -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
   
   **Вариант C: Через Python Launcher**
   ```bash
   # После установки Python 3.10 вручную:
   py -3.10 -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Для других систем:** Используйте pyenv или conda (см. [INSTALL.md](INSTALL.md))

⚠️ **Важно:** MediaPipe требует Python 3.8-3.11 (рекомендуется 3.10)

Подробнее см. [INSTALL.md](INSTALL.md) и [INSTALL_PYTHON_WINDOWS.md](INSTALL_PYTHON_WINDOWS.md)

## Использование

### Запуск с камеры (карты захвата)
1. Убедитесь, что карта захвата подключена
2. При необходимости измените `CAMERA_ID` в `config.py` (обычно 0 или 1)
3. Запустите:
```bash
python main.py
```

### Запуск с видео файла для отладки

Есть два способа запустить с видео файлом:

#### Способ 1: Через аргумент командной строки (рекомендуется)
```bash
python main.py -v "путь/к/видео.mp4"
```
или
```bash
python main.py --video "путь/к/видео.mp4"
```

Примеры:
```bash
python main.py -v test_video.mp4
python main.py -v videos/workout.mp4
python main.py -v "C:/Videos/barbell_side_view.mp4"
```

#### Способ 2: Через config.py
1. Откройте `config.py`
2. Установите `DEBUG_VIDEO_PATH = "путь/к/вашему/видео.mp4"`
3. Запустите:
```bash
python main.py
```

**Примечание:** Если указан аргумент `-v` или `--video`, он имеет приоритет над `DEBUG_VIDEO_PATH` в config.py.

### Управление (горячие клавиши)
- `q` — выход
- `c` — очистить путь штанги вручную
- `1` — полностью выключить трекинг позы (для экономии ресурсов)
- `l` — включить/выключить визуализацию ног (включает позу, если была выключена)
- `d` — включить/выключить режим отладки штанги (ROI/круги)

Настройка алгоритма HoughCircles на лету:
- `w`/`s` — поднять/опустить `param2` (основной порог)
- `e`/`r` — `param1` (порог Canny)
- `t`/`y` — `minRadius`
- `u`/`i` — `maxRadius`
- `o`/`p` — `minDist`
- `k`/`j` — уменьшить/увеличить размытие (size/sigma)

**Примечание:** При использовании видео файла программа автоматически остановится после завершения видео.

### Тестирование UDP отправки
Для проверки работы UDP отправки запустите тестовый получатель:
```bash
python test_udp_receiver.py
```
Затем запустите основной модуль. Тестовый скрипт будет выводить все получаемые данные в консоль.

## Структура проекта

- `main.py` — главный цикл, UDP, визуализация, OptimizedBarbellTracker
- `pose_tracker.py` — обертка над MediaPipe Pose (вычисление углов коленей, форматы данных)
- `barbell_tracker.py` — простой трекер штанги (эталон/пример)
- `visualizer.py` — отрисовка пути штанги и скелета
- `config.py` — конфигурация (все параметры приложения)
- `test_udp_receiver.py` — простой UDP‑приемник для отладки

## Формат данных UDP

JSON отправляется на `UDP_HOST:UDP_PORT` (по умолчанию 127.0.0.1:5005):

```json
{
  "timestamp": 1234567890.123,
  "barbell": {
    "position": [502, 298],           // текущая точка (px) или null
    "confidence": 0.82,               // оценка уверенности (0..1) или null
    "source": "hough"                 // "hough" | "contour" | "predict"
  },
  "knee_positions": {
    "left_knee": [320, 480],
    "right_knee": [600, 480],
    "left_knee_angle": 145.5,
    "right_knee_angle": 150.2
  },
  "joints": {
    "0": [0.5, 0.3, 0.0],
    "1": [0.52, 0.31, 0.0],
    ...
    "33": [0.6, 0.8, 0.0]
  },
  "barbell_path": [
    {"x": 500.0, "y": 300.0, "timestamp": 1234567890.100},
    {"x": 502.0, "y": 298.0, "timestamp": 1234567890.120}
  ]
}
```

Где:
- `barbell` — текущая точка штанги + метаданные
- `knee_positions` — координаты коленей (px) и углы
- `joints` — все 33 landmarks MediaPipe Pose (нормализованные 0..1: x, y, z)
- `barbell_path` — путь движения (px) с временными метками

## Ключевые настройки (`config.py`)

Видео/ввод:
- `VIDEO_WIDTH`, `VIDEO_HEIGHT`, `TARGET_FPS`, `CAMERA_ID`, `DEBUG_VIDEO_PATH`

UDP:
- `UDP_HOST`, `UDP_PORT`

Поза (опционально):
- `ENABLE_POSE_TRACKING`, `ENABLE_LEG_TRACKING`
- `MIN_DETECTION_CONFIDENCE`, `MIN_TRACKING_CONFIDENCE`

Штанга — детекция окружностей (Hough):
- `BARBELL_CIRCLE_MIN_RADIUS`, `BARBELL_CIRCLE_MAX_RADIUS`
- `BARBELL_CIRCLE_PARAM1`, `BARBELL_CIRCLE_PARAM2`, `BARBELL_CIRCLE_MIN_DIST`, `BARBELL_CIRCLE_DP`
- Fallback по контурам: `BARBELL_ENABLE_CONTOUR_FALLBACK`, `BARBELL_CANNY_THRESHOLD1/2`, `BARBELL_MIN_CIRCULARITY`

Штанга — устойчивость/без задержки:
- Поиск в ROI между запястьями: `BARBELL_USE_SEARCH_REGION`
- Быстрый анти‑джиттер без лага: `BARBELL_ANTI_JITTER*`, `BARBELL_X_JITTER_LOCK*`
- Выбор окружности с учетом стабильности по X: `BARBELL_X_STABILITY_GAIN`
- EMA/Калман (если нужно сглаживание ценой задержки): `BARBELL_SMOOTHING_FACTOR`, `BARBELL_USE_KALMAN`

Авто‑очистка пути при простое:
- `BARBELL_IDLE_CLEAR_SECONDS` — время простоя до очистки
- `BARBELL_IDLE_MIN_MOVE_PX` — минимальный сдвиг, чтобы считать «движение»

Визуализация:
- Цвета/толщина/радиусы, сглаживание рендеринга траектории: `PATH_SMOOTHING_*`

## Требования

- Python 3.8+
- OpenCV 4.8+
- MediaPipe 0.10+
- NumPy 1.24+

## Как это работает (кратко)

1) Видеокадры попадают в главный цикл `main.py`.
2) При включенной позе кадр отправляется в `PoseTracker`, который возвращает landmarks, координаты коленей и их углы. Запястья задают ROI для поиска штанги.
3) `OptimizedBarbellTracker` ищет круги (Hough), при необходимости — fallback по контурам. Затем выбирает «лучший» круг по стабильности и применяет анти‑джиттер без задержки.
4) Путь штанги пополняется новыми точками. Если штанга не двигается `BARBELL_IDLE_CLEAR_SECONDS`, путь очищается автоматически.
5) `Visualizer` рисует путь, текущую точку штанги и (опционально) скелет.
6) Все данные отправляются по UDP в JSON.

