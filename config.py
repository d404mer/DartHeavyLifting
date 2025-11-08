"""
Конфигурация параметров для отслеживания позы и штанги
"""

# Параметры видео
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
TARGET_FPS = 50

# Параметры UDP отправки
UDP_HOST = "127.0.0.1"
UDP_PORT = 5005

# Параметры MediaPipe Pose
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Параметры обнаружения штанги (окружность)
BARBELL_CIRCLE_MIN_RADIUS = 20
BARBELL_CIRCLE_MAX_RADIUS = 100
BARBELL_CIRCLE_PARAM1 = 50
BARBELL_CIRCLE_PARAM2 = 30

# Путь к видео файлу для отладки (None = использовать камеру)
# Можно указать относительный или абсолютный путь
# Примеры: "test_video.mp4", "videos/workout.mp4", "C:/Videos/barbell.mp4"
DEBUG_VIDEO_PATH = None  # Установите путь к видео файлу для отладки

# ID камеры (0 для первой камеры, обычно карта захвата будет с другим индексом)
CAMERA_ID = 0

# Цвета для визуализации
COLOR_BARBELL_PATH = (0, 255, 0)  # Зеленый для пути штанги
COLOR_KNEE_JOINT = (255, 0, 0)    # Синий для коленей
COLOR_LEG_BONE = (0, 0, 255)      # Красный для костей ног
COLOR_JOINT = (255, 255, 0)       # Голубой для других суставов

# Размер точек для визуализации
JOINT_RADIUS = 5
LINE_THICKNESS = 2

# Максимальное количество точек пути штанги для хранения
MAX_PATH_POINTS = 1000

