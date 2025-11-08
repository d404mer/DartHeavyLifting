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
    """Оптимизированный трекер штанги с привязкой к рукам спортсмена"""
    
    def __init__(self):
        self.path = deque(maxlen=config.MAX_PATH_POINTS)
        self.last_position = None
        self.frames_without_detection = 0
        self.search_region = None  # Область поиска на основе положения рук
        
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
    
    def detect_barbell(self, frame: np.ndarray, timestamp: float) -> Optional[Tuple[int, int]]:
        """
        Обнаружение штанги с оптимизацией
        
        Args:
            frame: Входной кадр (BGR)
            timestamp: Временная метка
            
        Returns:
            Координаты центра штанги (x, y) или None
        """
        # Используем область поиска если доступна
        if self.search_region:
            x, y, w, h = self.search_region
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0:
                self.search_region = None
                return None
        else:
            roi = frame
            x, y = 0, 0
        
        # Конвертация в grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Применение размытия для уменьшения шума
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Обнаружение окружностей
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=config.BARBELL_CIRCLE_PARAM1,
            param2=config.BARBELL_CIRCLE_PARAM2,
            minRadius=config.BARBELL_CIRCLE_MIN_RADIUS,
            maxRadius=config.BARBELL_CIRCLE_MAX_RADIUS
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            # Выбираем лучшую окружность
            best_circle = self._select_best_circle(circles)
            
            if best_circle is not None:
                cx, cy, r = best_circle
                # Преобразуем координаты обратно в координаты полного кадра
                global_x = cx + x
                global_y = cy + y
                
                self.last_position = (global_x, global_y)
                self.frames_without_detection = 0
                self.path.append((global_x, global_y, timestamp))
                
                return (global_x, global_y)
        
        # Штанга не обнаружена
        self.frames_without_detection += 1
        return None
    
    def _select_best_circle(self, circles: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """
        Выбор лучшей окружности из найденных
        
        Args:
            circles: Массив окружностей [(x, y, r), ...]
            
        Returns:
            Лучшая окружность (x, y, r) или None
        """
        if len(circles) == 0:
            return None
        
        if len(circles) == 1:
            return tuple(circles[0])
        
        # Если есть последнее известное положение - выбираем ближайшую
        if self.last_position is not None:
            distances = []
            for circle in circles:
                cx, cy, r = circle
                dist = np.sqrt((cx - self.last_position[0])**2 + (cy - self.last_position[1])**2)
                distances.append(dist)
            closest_idx = np.argmin(distances)
            return tuple(circles[closest_idx])
        
        # Иначе выбираем окружность с самым большим радиусом (скорее всего штанга)
        largest_idx = np.argmax([c[2] for c in circles])
        return tuple(circles[largest_idx])
    
    def get_path(self) -> List[Tuple[float, float, float]]:
        """Получение пути штанги"""
        return list(self.path)
    
    def clear_path(self):
        """Очистка пути"""
        self.path.clear()
        self.last_position = None
        self.frames_without_detection = 0


# -------------------------
# Скелетные соединения (как в примере)
# -------------------------
skeleton_connections = [
    # Торс
    (11, 12), (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), 
    (25, 27), (27, 29), (26, 28), (28, 30),
    # Руки (только плечо-локоть-запястье)
    (11, 13), (13, 15),
    (12, 14), (14, 16)
]

# Точки для отображения (как в примере, но добавим колени)
display_points = [11, 13, 15, 12, 14, 16, 23, 24, 25, 26, 27, 28, 29, 30]

# Индексы MediaPipe Pose
LEFT_HIP = 23
LEFT_KNEE = 25
LEFT_ANKLE = 27
RIGHT_HIP = 24
RIGHT_KNEE = 26
RIGHT_ANKLE = 28
LEFT_WRIST = 15
RIGHT_WRIST = 16


# -------------------------
# Основной цикл
# -------------------------
def main():
    """Главная функция"""
    # Парсинг аргументов
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
    
    print("Запуск отслеживания. Нажмите 'q' для выхода, 'c' для очистки пути штанги")
    
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
            
            # Конвертация в RGB для MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            
            # Обработка позы
            results = pose.process(frame_rgb)
            
            # Конвертация обратно для отображения
            frame_rgb.flags.writeable = True
            frame_display = frame.copy()
            
            if results.pose_landmarks:
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
                
                # Получаем координаты запястий для трекинга штанги
                left_wrist_px = points_px[LEFT_WRIST] if lm[LEFT_WRIST].visibility > 0.5 else None
                right_wrist_px = points_px[RIGHT_WRIST] if lm[RIGHT_WRIST].visibility > 0.5 else None
                
                # Вычисление углов коленей
                left_knee_angle = None
                right_knee_angle = None
                
                if all([left_hip, left_knee, left_ankle]):
                    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                
                if all([right_hip, right_knee, right_ankle]):
                    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
                
                # Обновляем область поиска штанги на основе положения рук
                barbell_tracker.update_search_region(left_wrist_px, right_wrist_px, frame.shape)
                
                # Обнаружение штанги
                barbell_pos = barbell_tracker.detect_barbell(frame, timestamp)
                
                # Подготовка данных для UDP
                joints_data = {str(i): [float(l.x), float(l.y), float(getattr(l, 'z', 0.0))] 
                              for i, l in enumerate(lm)}
                
                # Отправка данных через UDP
                udp_data = {
                    "timestamp": timestamp,
                    "knee_positions": {
                        "left_knee": list(points_px[LEFT_KNEE]) if left_knee else None,
                        "right_knee": list(points_px[RIGHT_KNEE]) if right_knee else None,
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
                
                # Визуализация пути штанги
                barbell_path = barbell_tracker.get_path()
                if len(barbell_path) > 1:
                    for i in range(1, len(barbell_path)):
                        x1, y1, _ = barbell_path[i-1]
                        x2, y2, _ = barbell_path[i]
                        cv2.line(frame_display, (int(x1), int(y1)), (int(x2), int(y2)),
                                config.COLOR_BARBELL_PATH, config.LINE_THICKNESS)
                
                # Рисуем текущее положение штанги
                if barbell_pos:
                    bx, by = barbell_pos
                    cv2.circle(frame_display, (int(bx), int(by)), 15, config.COLOR_BARBELL_PATH, 3)
                    cv2.line(frame_display, (int(bx)-10, int(by)), (int(bx)+10, int(by)),
                            config.COLOR_BARBELL_PATH, 2)
                    cv2.line(frame_display, (int(bx), int(by)-10), (int(bx), int(by)+10),
                            config.COLOR_BARBELL_PATH, 2)
            
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
