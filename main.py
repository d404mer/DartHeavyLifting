"""
Главный модуль для запуска отслеживания позы и штанги
Основан на примере MediaPipe, оптимизирован и расширен
"""

import cv2
import mediapipe as mp
import numpy as np
import socket
import json
import time
import argparse
import os
import sys
from typing import Optional, Tuple, List
from collections import deque

import config


# -------------------------
# Настройка MediaPipe Pose
# -------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
    model_complexity=2
)

# -------------------------
# Настройка UDP
# -------------------------
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
UE_IP, UE_PORT = config.UDP_HOST, config.UDP_PORT

# -------------------------
# Функция для вычисления угла (как в примере)
# -------------------------
def calculate_angle(a, b, c):
    """Расчет угла между тремя точками"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# -------------------------
# Оптимизированный трекер штанги
# -------------------------
class OptimizedBarbellTracker:
    """Оптимизированный трекер штанги с привязкой к рукам спортсмена и сглаживанием"""
    
    def __init__(self, smoothing_factor=0.7):
        """
        Args:
            smoothing_factor: Коэффициент сглаживания (0-1), чем больше - тем плавнее, но больше задержка
        """
        self.path = deque(maxlen=config.MAX_PATH_POINTS)
        self.last_position = None
        self.smoothed_position = None  # Сглаженное положение
        self.frames_without_detection = 0
        self.search_region = None  # Область поиска на основе положения рук
        self.smoothing_factor = smoothing_factor  # Коэффициент экспоненциального сглаживания
        self.last_radius = None  # Последний радиус для стабильности выбора
        
    def update_search_region(self, left_wrist, right_wrist, frame_shape):
        """
        Обновление области поиска на основе положения рук
        
        Args:
            left_wrist: Координаты левого запястья (x, y) в пикселях
            right_wrist: Координаты правого запястья (x, y) в пикселях
            frame_shape: Размеры кадра (height, width)
        """
        if left_wrist and right_wrist:
            # Область между руками и немного выше
            min_x = min(left_wrist[0], right_wrist[0]) - 100
            max_x = max(left_wrist[0], right_wrist[0]) + 100
            min_y = min(left_wrist[1], right_wrist[1]) - 150
            max_y = max(left_wrist[1], right_wrist[1]) + 50
            
            # Ограничиваем границами кадра
            h, w = frame_shape[:2]
            min_x = max(0, min_x)
            max_x = min(w, max_x)
            min_y = max(0, min_y)
            max_y = min(h, max_y)
            
            self.search_region = (int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y))
        else:
            self.search_region = None
    
    def detect_barbell(self, frame: np.ndarray, timestamp: float, debug_frame=None) -> Optional[Tuple[int, int]]:
        """
        Обнаружение штанги с оптимизацией
        
        Args:
            frame: Входной кадр (BGR)
            timestamp: Временная метка
            debug_frame: Кадр для отладки (если нужна визуализация процесса)
            
        Returns:
            Координаты центра штанги (x, y) или None
        """
        # Используем область поиска если доступна и включена
        if config.BARBELL_USE_SEARCH_REGION and self.search_region:
            x, y, w, h = self.search_region
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0:
                self.search_region = None
                return None
            # Рисуем область поиска в режиме отладки
            if config.BARBELL_DEBUG_MODE and debug_frame is not None:
                cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
        else:
            roi = frame
            x, y = 0, 0
        
        # Конвертация в grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Применение размытия для уменьшения шума
        blur_size = config.BARBELL_BLUR_SIZE
        blur_sigma = config.BARBELL_BLUR_SIGMA
        blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), blur_sigma)
        
        # Визуализация размытого изображения в режиме отладки
        # Черно-белый прямоугольник = область поиска после предобработки (grayscale + размытие)
        # Это то изображение, которое анализирует алгоритм HoughCircles для поиска окружностей
        if config.BARBELL_DEBUG_MODE and debug_frame is not None:
            debug_blur = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
            if x > 0 or y > 0:
                # Показываем размытое изображение только если используется область поиска
                debug_frame[y:y+debug_blur.shape[0], x:x+debug_blur.shape[1]] = debug_blur
        
        # Обнаружение окружностей
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=config.BARBELL_CIRCLE_DP,
            minDist=config.BARBELL_CIRCLE_MIN_DIST,
            param1=config.BARBELL_CIRCLE_PARAM1,
            param2=config.BARBELL_CIRCLE_PARAM2,
            minRadius=config.BARBELL_CIRCLE_MIN_RADIUS,
            maxRadius=config.BARBELL_CIRCLE_MAX_RADIUS
        )
        
        # Визуализация всех найденных окружностей в режиме отладки
        if config.BARBELL_DEBUG_MODE and debug_frame is not None and circles is not None:
            circles_int = np.round(circles[0, :]).astype("int")
            for (cx, cy, r) in circles_int:
                global_cx = cx + x
                global_cy = cy + y
                cv2.circle(debug_frame, (global_cx, global_cy), r, (0, 255, 255), 2)
                cv2.circle(debug_frame, (global_cx, global_cy), 2, (0, 255, 255), -1)
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            # Выбираем лучшую окружность
            best_circle = self._select_best_circle(circles)
            
            if best_circle is not None:
                cx, cy, r = best_circle
                # Преобразуем координаты обратно в координаты полного кадра
                global_x = cx + x
                global_y = cy + y
                
                # Применяем сглаживание
                if self.smoothed_position is None:
                    self.smoothed_position = (float(global_x), float(global_y))
                else:
                    # Экспоненциальное сглаживание
                    smooth_x = self.smoothing_factor * self.smoothed_position[0] + (1 - self.smoothing_factor) * global_x
                    smooth_y = self.smoothing_factor * self.smoothed_position[1] + (1 - self.smoothing_factor) * global_y
                    self.smoothed_position = (smooth_x, smooth_y)
                
                self.last_position = (global_x, global_y)
                self.frames_without_detection = 0
                
                # Добавляем сглаженное положение в путь
                self.path.append((self.smoothed_position[0], self.smoothed_position[1], timestamp))
                
                return (int(self.smoothed_position[0]), int(self.smoothed_position[1]))
        
        # Штанга не обнаружена
        self.frames_without_detection += 1
        
        # Если есть сглаженное положение и прошло мало кадров - используем его (предсказание)
        if self.smoothed_position is not None and self.frames_without_detection < 5:
            # Используем последнее сглаженное положение с небольшим сдвигом
            # (простое предсказание на основе последнего движения)
            if len(self.path) >= 2:
                # Вычисляем скорость на основе последних двух точек
                prev_x, prev_y, prev_ts = self.path[-1]
                prev2_x, prev2_y, prev2_ts = self.path[-2]
                dt = prev_ts - prev2_ts
                if dt > 0:
                    vx = (prev_x - prev2_x) / dt
                    vy = (prev_y - prev2_y) / dt
                    # Предсказываем следующее положение
                    predicted_x = prev_x + vx * (timestamp - prev_ts)
                    predicted_y = prev_y + vy * (timestamp - prev_ts)
                    self.path.append((predicted_x, predicted_y, timestamp))
                    return (int(predicted_x), int(predicted_y))
            else:
                # Просто используем последнее сглаженное положение
                return (int(self.smoothed_position[0]), int(self.smoothed_position[1]))
        
        return None
    
    def _select_best_circle(self, circles: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """
        Улучшенный выбор лучшей окружности из найденных
        
        Args:
            circles: Массив окружностей [(x, y, r), ...]
            
        Returns:
            Лучшая окружность (x, y, r) или None
        """
        if len(circles) == 0:
            return None
        
        if len(circles) == 1:
            return tuple(circles[0])
        
        # Если есть последнее известное положение - используем улучшенный алгоритм
        if self.last_position is not None and self.smoothed_position is not None:
            best_circle = None
            best_score = -1
            
            last_x, last_y = self.last_position
            last_radius = None
            if hasattr(self, 'last_radius'):
                last_radius = self.last_radius
            
            for circle in circles:
                cx, cy, r = circle
                
                # Оценка по расстоянию до последней позиции
                position_dist = np.sqrt((cx - last_x)**2 + (cy - last_y)**2)
                position_score = 1.0 / (1.0 + position_dist / 100.0)  # Нормализуем
                
                # Оценка по стабильности радиуса
                radius_score = 1.0
                if last_radius is not None:
                    radius_diff = abs(r - last_radius)
                    radius_score = 1.0 / (1.0 + radius_diff / 10.0)  # Предпочитаем похожий радиус
                
                # Оценка по размеру радиуса (если включено)
                size_score = 1.0
                if config.BARBELL_PREFER_LARGER_RADIUS:
                    # Нормализуем радиус относительно диапазона
                    radius_range = config.BARBELL_CIRCLE_MAX_RADIUS - config.BARBELL_CIRCLE_MIN_RADIUS
                    if radius_range > 0:
                        normalized_radius = (r - config.BARBELL_CIRCLE_MIN_RADIUS) / radius_range
                        size_score = normalized_radius * 0.3 + 0.7  # Небольшой бонус за больший радиус
                
                # Комбинированный score
                combined_score = (
                    position_score * config.BARBELL_POSITION_STABILITY_WEIGHT +
                    radius_score * config.BARBELL_RADIUS_STABILITY_WEIGHT +
                    size_score * (1.0 - config.BARBELL_POSITION_STABILITY_WEIGHT - config.BARBELL_RADIUS_STABILITY_WEIGHT)
                )
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_circle = circle
            
            if best_circle is not None:
                # Сохраняем радиус для следующего кадра
                self.last_radius = best_circle[2]
                return tuple(best_circle)
        
        # Иначе выбираем по размеру радиуса или случайную
        if config.BARBELL_PREFER_LARGER_RADIUS:
            largest_idx = np.argmax([c[2] for c in circles])
            self.last_radius = circles[largest_idx][2]
            return tuple(circles[largest_idx])
        else:
            # Выбираем средний радиус
            radii = [c[2] for c in circles]
            avg_radius = np.mean(radii)
            closest_to_avg_idx = np.argmin([abs(c[2] - avg_radius) for c in circles])
            self.last_radius = circles[closest_to_avg_idx][2]
            return tuple(circles[closest_to_avg_idx])
    
    def get_path(self) -> List[Tuple[float, float, float]]:
        """Получение пути штанги"""
        return list(self.path)
    
    def clear_path(self):
        """Очистка пути"""
        self.path.clear()
        self.last_position = None
        self.smoothed_position = None
        self.frames_without_detection = 0



# Соединения только для ног (торс + ноги, без рук)
skeleton_connections = [
    # Торс (только для соединения с ногами)
    (23, 24),  # Соединение бедер
    # Левая нога
    (23, 25), (25, 27), (27, 29), (29, 31),
    # Правая нога
    (24, 26), (26, 28), (28, 30), (30, 32)
]

# Точки для отображения - только ноги
display_points = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]  # Бедра, колени, лодыжки, стопы

# Индексы MediaPipe Pose
LEFT_HIP = 23
LEFT_KNEE = 25
LEFT_ANKLE = 27
RIGHT_HIP = 24
RIGHT_KNEE = 26
RIGHT_ANKLE = 28
LEFT_WRIST = 15
RIGHT_WRIST = 16



def main():
    """Главная функция"""
 
    parser = argparse.ArgumentParser(description='Отслеживание позы и штанги')
    parser.add_argument('-v', '--video', type=str, default=None,
                       help='Путь к видео файлу (если не указан, используется камера)')
    args = parser.parse_args()
    
    # Инициализация источника видео
    video_path = args.video if args.video else config.DEBUG_VIDEO_PATH
    
    if video_path:
        if not os.path.exists(video_path):
            print(f"Ошибка: видео файл не найден: {video_path}")
            return
        print(f"Использование видео файла: {video_path}")
        cap = cv2.VideoCapture(video_path)
    else:
        print(f"Использование камеры с ID: {config.CAMERA_ID}")
        cap = cv2.VideoCapture(config.CAMERA_ID)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.VIDEO_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.VIDEO_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, config.TARGET_FPS)
    
    if not cap.isOpened():
        print("Ошибка: не удалось открыть источник видео")
        return
    
    # Инициализация трекера штанги
    barbell_tracker = OptimizedBarbellTracker()
    
    print("Запуск отслеживания.")
    print("Горячие клавиши:")
    print("  === Основные ===")
    print("  'q' - выход")
    print("  'c' - очистить путь штанги")
    print("  '1' - полностью отключить трекинг тела (экономия ресурсов)")
    print("  'l' - переключить трекинг ног (включает трекинг тела если выключен)")
    print("  'd' - переключить режим отладки")
    print("  === Настройка штанги ===")
    print("  'w'/'s' - увеличить/уменьшить param2 (основной параметр)")
    print("  'e'/'r' - увеличить/уменьшить param1")
    print("  't'/'y' - уменьшить/увеличить minRadius")
    print("  'u'/'i' - уменьшить/увеличить maxRadius")
    print("  'o'/'p' - уменьшить/увеличить minDist")
    print("  'k'/'j' - уменьшить/увеличить размытие")
    if config.BARBELL_DEBUG_MODE:
        print("\n[РЕЖИМ ОТЛАДКИ ВКЛЮЧЕН]")
        print("Желтый прямоугольник - граница области поиска (между руками)")
        print("Черно-белый прямоугольник - область поиска после предобработки (grayscale + размытие)")
        print("  Это изображение анализирует алгоритм для поиска окружностей")
        print("Желтые окружности - все найденные окружности (алгоритм HoughCircles)")
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                if video_path:
                    print("Конец видео файла")
                    break
                continue
            
            h, w = frame.shape[:2]
            timestamp = time.time()
            
            # Конвертация обратно для отображения
            frame_display = frame.copy()
            
            # Обработка позы только если включен трекинг позы
            results = None
            if config.ENABLE_POSE_TRACKING:
                # Конвертация в RGB для MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False
                
                # Обработка позы
                results = pose.process(frame_rgb)
                
                # Конвертация обратно для отображения
                frame_rgb.flags.writeable = True
            
            if results and results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                
                # Получаем координаты в пикселях
                points_px = [(int(l.x * w), int(l.y * h)) for l in lm]
                
                # Получаем координаты ключевых точек
                left_hip = (lm[LEFT_HIP].x, lm[LEFT_HIP].y) if lm[LEFT_HIP].visibility > 0.5 else None
                left_knee = (lm[LEFT_KNEE].x, lm[LEFT_KNEE].y) if lm[LEFT_KNEE].visibility > 0.5 else None
                left_ankle = (lm[LEFT_ANKLE].x, lm[LEFT_ANKLE].y) if lm[LEFT_ANKLE].visibility > 0.5 else None
                
                right_hip = (lm[RIGHT_HIP].x, lm[RIGHT_HIP].y) if lm[RIGHT_HIP].visibility > 0.5 else None
                right_knee = (lm[RIGHT_KNEE].x, lm[RIGHT_KNEE].y) if lm[RIGHT_KNEE].visibility > 0.5 else None
                right_ankle = (lm[RIGHT_ANKLE].x, lm[RIGHT_ANKLE].y) if lm[RIGHT_ANKLE].visibility > 0.5 else None
                
                # Получаем координаты запястий для трекинга штанги (если нужно)
                left_wrist_px = None
                right_wrist_px = None
                if config.BARBELL_USE_SEARCH_REGION:
                    left_wrist_px = points_px[LEFT_WRIST] if lm[LEFT_WRIST].visibility > 0.5 else None
                    right_wrist_px = points_px[RIGHT_WRIST] if lm[RIGHT_WRIST].visibility > 0.5 else None
                
                # Вычисление углов коленей (только если включен трекинг ног)
                left_knee_angle = None
                right_knee_angle = None
                
                if config.ENABLE_LEG_TRACKING:
                    if all([left_hip, left_knee, left_ankle]):
                        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                    
                    if all([right_hip, right_knee, right_ankle]):
                        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
                
                # Обновляем область поиска штанги на основе положения рук
                if config.BARBELL_USE_SEARCH_REGION and (left_wrist_px or right_wrist_px):
                    barbell_tracker.update_search_region(left_wrist_px, right_wrist_px, frame.shape)
            else:
                # Если трекинг позы выключен, отключаем область поиска на основе рук
                if config.BARBELL_USE_SEARCH_REGION:
                    barbell_tracker.search_region = None
            
            # Обнаружение штанги (всегда работает, независимо от трекинга позы)
            barbell_pos = barbell_tracker.detect_barbell(frame, timestamp, debug_frame=frame_display if config.BARBELL_DEBUG_MODE else None)
            
            # Подготовка данных для UDP
            if config.ENABLE_POSE_TRACKING and results and results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                points_px = [(int(l.x * w), int(l.y * h)) for l in lm]
                
                # Получаем координаты ключевых точек для UDP
                left_knee_coords = list(points_px[LEFT_KNEE]) if (lm[LEFT_KNEE].visibility > 0.5 and config.ENABLE_LEG_TRACKING) else None
                right_knee_coords = list(points_px[RIGHT_KNEE]) if (lm[RIGHT_KNEE].visibility > 0.5 and config.ENABLE_LEG_TRACKING) else None
                
                # Подготовка joints данных
                joints_data = {str(i): [float(l.x), float(l.y), float(getattr(l, 'z', 0.0))] 
                              for i, l in enumerate(lm)} if config.ENABLE_LEG_TRACKING else {}
                
                # Получаем углы коленей
                left_hip = (lm[LEFT_HIP].x, lm[LEFT_HIP].y) if lm[LEFT_HIP].visibility > 0.5 else None
                left_knee = (lm[LEFT_KNEE].x, lm[LEFT_KNEE].y) if lm[LEFT_KNEE].visibility > 0.5 else None
                left_ankle = (lm[LEFT_ANKLE].x, lm[LEFT_ANKLE].y) if lm[LEFT_ANKLE].visibility > 0.5 else None
                right_hip = (lm[RIGHT_HIP].x, lm[RIGHT_HIP].y) if lm[RIGHT_HIP].visibility > 0.5 else None
                right_knee = (lm[RIGHT_KNEE].x, lm[RIGHT_KNEE].y) if lm[RIGHT_KNEE].visibility > 0.5 else None
                right_ankle = (lm[RIGHT_ANKLE].x, lm[RIGHT_ANKLE].y) if lm[RIGHT_ANKLE].visibility > 0.5 else None
                
                left_knee_angle = None
                right_knee_angle = None
                if config.ENABLE_LEG_TRACKING:
                    if all([left_hip, left_knee, left_ankle]):
                        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                    if all([right_hip, right_knee, right_ankle]):
                        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
            else:
                # Если трекинг позы выключен - пустые данные
                left_knee_coords = None
                right_knee_coords = None
                left_knee_angle = None
                right_knee_angle = None
                joints_data = {}
            
            # Отправка данных через UDP
            udp_data = {
                "timestamp": timestamp,
                "knee_positions": {
                    "left_knee": left_knee_coords,
                    "right_knee": right_knee_coords,
                    "left_knee_angle": float(left_knee_angle) if left_knee_angle else None,
                    "right_knee_angle": float(right_knee_angle) if right_knee_angle else None
                },
                "joints": joints_data,
                "barbell_path": [
                    {"x": float(x), "y": float(y), "timestamp": float(ts)}
                    for x, y, ts in barbell_tracker.get_path()
                ]
            }
            
            try:
                sock.sendto(json.dumps(udp_data, ensure_ascii=False).encode('utf-8'), (UE_IP, UE_PORT))
            except Exception as e:
                print(f"Ошибка отправки UDP: {e}")
            
            # Визуализация скелета (только если включен трекинг позы и ног)
            if config.ENABLE_POSE_TRACKING and results and results.pose_landmarks and config.ENABLE_LEG_TRACKING:
                lm = results.pose_landmarks.landmark
                points_px = [(int(l.x * w), int(l.y * h)) for l in lm]
                
                # Получаем углы коленей для визуализации
                left_hip = (lm[LEFT_HIP].x, lm[LEFT_HIP].y) if lm[LEFT_HIP].visibility > 0.5 else None
                left_knee = (lm[LEFT_KNEE].x, lm[LEFT_KNEE].y) if lm[LEFT_KNEE].visibility > 0.5 else None
                left_ankle = (lm[LEFT_ANKLE].x, lm[LEFT_ANKLE].y) if lm[LEFT_ANKLE].visibility > 0.5 else None
                right_hip = (lm[RIGHT_HIP].x, lm[RIGHT_HIP].y) if lm[RIGHT_HIP].visibility > 0.5 else None
                right_knee = (lm[RIGHT_KNEE].x, lm[RIGHT_KNEE].y) if lm[RIGHT_KNEE].visibility > 0.5 else None
                right_ankle = (lm[RIGHT_ANKLE].x, lm[RIGHT_ANKLE].y) if lm[RIGHT_ANKLE].visibility > 0.5 else None
                
                left_knee_angle = None
                right_knee_angle = None
                if all([left_hip, left_knee, left_ankle]):
                    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                if all([right_hip, right_knee, right_ankle]):
                    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
                
                # Визуализация скелета
                for start_idx, end_idx in skeleton_connections:
                    if (lm[start_idx].visibility > 0.5 and lm[end_idx].visibility > 0.5):
                        cv2.line(frame_display, points_px[start_idx], points_px[end_idx], 
                                config.COLOR_LEG_BONE, config.LINE_THICKNESS)
                
                # Рисуем точки суставов
                for idx in display_points:
                    if lm[idx].visibility > 0.5:
                        point = points_px[idx]
                        # Колени выделяем другим цветом
                        if idx in [LEFT_KNEE, RIGHT_KNEE]:
                            cv2.circle(frame_display, point, config.JOINT_RADIUS + 2, 
                                      config.COLOR_KNEE_JOINT, -1)
                            # Отображаем угол колена
                            if idx == LEFT_KNEE and left_knee_angle:
                                cv2.putText(frame_display, f"{int(left_knee_angle)}°",
                                           (point[0] + 15, point[1] - 15),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_KNEE_JOINT, 2)
                            elif idx == RIGHT_KNEE and right_knee_angle:
                                cv2.putText(frame_display, f"{int(right_knee_angle)}°",
                                           (point[0] + 15, point[1] - 15),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_KNEE_JOINT, 2)
                        else:
                            cv2.circle(frame_display, point, config.JOINT_RADIUS, 
                                      config.COLOR_JOINT, -1)
            
            # Визуализация пути штанги (всегда, независимо от трекинга позы)
            barbell_path = barbell_tracker.get_path()
            if len(barbell_path) > 1:
                # Рисуем сглаженные линии между точками
                smoothed_points = []
                for i, (x, y, ts) in enumerate(barbell_path):
                    if i == 0 or i == len(barbell_path) - 1:
                        smoothed_points.append((int(x), int(y)))
                    else:
                        # Простое сглаживание - среднее между соседними точками
                        prev_x, prev_y, _ = barbell_path[i-1]
                        next_x, next_y, _ = barbell_path[i+1]
                        smooth_x = (prev_x + x + next_x) / 3
                        smooth_y = (prev_y + y + next_y) / 3
                        smoothed_points.append((int(smooth_x), int(smooth_y)))
                
                # Рисуем сглаженные линии
                for i in range(1, len(smoothed_points)):
                    cv2.line(frame_display, smoothed_points[i-1], smoothed_points[i],
                            config.COLOR_BARBELL_PATH, config.LINE_THICKNESS)
            
            # Отображение результата
            cv2.imshow('Barbell & Pose Tracking', frame_display)
            
            # Обработка клавиш
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Выход по запросу пользователя")
                break
            elif key == ord('c'):
                barbell_tracker.clear_path()
                print("Путь штанги очищен")
            elif key == ord('1'):
                config.ENABLE_POSE_TRACKING = False
                config.ENABLE_LEG_TRACKING = False
                print("Трекинг тела: ВЫКЛ (полностью отключен)")
            elif key == ord('l') or key == ord('L'):
                if not config.ENABLE_POSE_TRACKING:
                    # Если трекинг позы выключен, сначала включаем его
                    config.ENABLE_POSE_TRACKING = True
                    config.ENABLE_LEG_TRACKING = True
                    print("Трекинг тела: ВКЛ, Трекинг ног: ВКЛ")
                else:
                    config.ENABLE_LEG_TRACKING = not config.ENABLE_LEG_TRACKING
                    print(f"Трекинг ног: {'ВКЛ' if config.ENABLE_LEG_TRACKING else 'ВЫКЛ'}")
            elif key == ord('d') or key == ord('D'):
                config.BARBELL_DEBUG_MODE = not config.BARBELL_DEBUG_MODE
                print(f"Режим отладки: {'ВКЛ' if config.BARBELL_DEBUG_MODE else 'ВЫКЛ'}")
            # Настройка param2 (основной параметр) - w/s
            elif key == ord('w') or key == ord('W'):
                config.BARBELL_CIRCLE_PARAM2 = min(50, config.BARBELL_CIRCLE_PARAM2 + 2)
                print(f"param2 (строже): {config.BARBELL_CIRCLE_PARAM2}")
            elif key == ord('s') or key == ord('S'):
                config.BARBELL_CIRCLE_PARAM2 = max(10, config.BARBELL_CIRCLE_PARAM2 - 2)
                print(f"param2 (мягче): {config.BARBELL_CIRCLE_PARAM2}")
            # Настройка param1 - e/r
            elif key == ord('e') or key == ord('E'):
                config.BARBELL_CIRCLE_PARAM1 = max(30, config.BARBELL_CIRCLE_PARAM1 - 10)
                print(f"param1: {config.BARBELL_CIRCLE_PARAM1}")
            elif key == ord('r') or key == ord('R'):
                config.BARBELL_CIRCLE_PARAM1 = min(200, config.BARBELL_CIRCLE_PARAM1 + 10)
                print(f"param1: {config.BARBELL_CIRCLE_PARAM1}")
            # Настройка minRadius - t/y
            elif key == ord('t') or key == ord('T'):
                config.BARBELL_CIRCLE_MIN_RADIUS = max(5, config.BARBELL_CIRCLE_MIN_RADIUS - 5)
                print(f"minRadius: {config.BARBELL_CIRCLE_MIN_RADIUS}")
            elif key == ord('y') or key == ord('Y'):
                config.BARBELL_CIRCLE_MIN_RADIUS = min(100, config.BARBELL_CIRCLE_MIN_RADIUS + 5)
                print(f"minRadius: {config.BARBELL_CIRCLE_MIN_RADIUS}")
            # Настройка maxRadius - u/i
            elif key == ord('u') or key == ord('U'):
                config.BARBELL_CIRCLE_MAX_RADIUS = max(20, config.BARBELL_CIRCLE_MAX_RADIUS - 10)
                print(f"maxRadius: {config.BARBELL_CIRCLE_MAX_RADIUS}")
            elif key == ord('i') or key == ord('I'):
                config.BARBELL_CIRCLE_MAX_RADIUS = min(200, config.BARBELL_CIRCLE_MAX_RADIUS + 10)
                print(f"maxRadius: {config.BARBELL_CIRCLE_MAX_RADIUS}")
            # Настройка minDist - o/p
            elif key == ord('o') or key == ord('O'):
                config.BARBELL_CIRCLE_MIN_DIST = max(20, config.BARBELL_CIRCLE_MIN_DIST - 10)
                print(f"minDist: {config.BARBELL_CIRCLE_MIN_DIST}")
            elif key == ord('p') or key == ord('P'):
                config.BARBELL_CIRCLE_MIN_DIST = min(200, config.BARBELL_CIRCLE_MIN_DIST + 10)
                print(f"minDist: {config.BARBELL_CIRCLE_MIN_DIST}")
            # Настройка размытия - k/j
            elif key == ord('k') or key == ord('K'):
                # Уменьшить размытие
                if config.BARBELL_BLUR_SIZE > 5:
                    config.BARBELL_BLUR_SIZE = max(5, config.BARBELL_BLUR_SIZE - 2)
                config.BARBELL_BLUR_SIGMA = max(1, config.BARBELL_BLUR_SIGMA - 0.5)
                print(f"Размытие: size={config.BARBELL_BLUR_SIZE}, sigma={config.BARBELL_BLUR_SIGMA:.1f}")
            elif key == ord('j') or key == ord('J'):
                # Увеличить размытие
                if config.BARBELL_BLUR_SIZE < 15:
                    config.BARBELL_BLUR_SIZE = min(15, config.BARBELL_BLUR_SIZE + 2)
                config.BARBELL_BLUR_SIGMA = min(5, config.BARBELL_BLUR_SIGMA + 0.5)
                print(f"Размытие: size={config.BARBELL_BLUR_SIZE}, sigma={config.BARBELL_BLUR_SIGMA:.1f}")
    
    except KeyboardInterrupt:
        print("\nПрерывание по Ctrl+C")
    
    finally:
        # Завершение работы
        cap.release()
        cv2.destroyAllWindows()
        sock.close()
        pose.close()
        print("Ресурсы освобождены")


if __name__ == "__main__":
    main()
