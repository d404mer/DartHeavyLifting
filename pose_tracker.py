"""
Модуль отслеживания позы человека с использованием MediaPipe
Отслеживает позу полностью через MediaPipe
"""

import mediapipe as mp
import numpy as np
import cv2
import logging
from typing import Optional, Dict, Tuple, List

logger = logging.getLogger(__name__)


class PoseTracker:
    """Класс для отслеживания позы человека через MediaPipe"""
    
    # Индексы ключевых точек MediaPipe Pose
    # Ноги
    LEFT_HIP = 23
    LEFT_KNEE = 25
    LEFT_ANKLE = 27
    LEFT_FOOT_INDEX = 31
    
    RIGHT_HIP = 24
    RIGHT_KNEE = 26
    RIGHT_ANKLE = 28
    RIGHT_FOOT_INDEX = 32
    
    # Руки (если понадобятся)
    LEFT_SHOULDER = 11
    LEFT_ELBOW = 13
    LEFT_WRIST = 15
    
    RIGHT_SHOULDER = 12
    RIGHT_ELBOW = 14
    RIGHT_WRIST = 16
    
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Инициализация трекера позы
        
        Args:
            min_detection_confidence: Минимальная уверенность для обнаружения
            min_tracking_confidence: Минимальная уверенность для отслеживания
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=2  # Полная модель для лучшей точности
        )
        
        # Соединения скелета для визуализации (торс и ноги, без лица и кистей)
        self.skeleton_connections = [
            # Торс
            (11, 12), (11, 23), (12, 24), (23, 24),
            # Левая нога
            (23, 25), (25, 27), (27, 29), (29, 31),
            # Правая нога
            (24, 26), (26, 28), (28, 30), (30, 32),
            # Руки (опционально, только основные точки)
            (11, 13), (13, 15),
            (12, 14), (14, 16)
        ]
        
        self.last_landmarks = None
        self.frame_width = 0
        self.frame_height = 0
        
    def process_frame(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Обработка кадра для обнаружения позы
        
        Args:
            frame: Входной кадр (BGR)
            
        Returns:
            Словарь с данными о позе или None если не обнаружено
        """
        # Сохраняем размеры кадра
        self.frame_height, self.frame_width = frame.shape[:2]
        
        # Конвертация BGR в RGB для MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        # Обработка позы
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            logger.debug("Поза не обнаружена")
            self.last_landmarks = None
            return None
        
        self.last_landmarks = results.pose_landmarks
        lm = results.pose_landmarks.landmark
        
        # Получаем координаты всех точек для ног
        left_hip = self._get_landmark_normalized(lm, self.LEFT_HIP)
        left_knee = self._get_landmark_normalized(lm, self.LEFT_KNEE)
        left_ankle = self._get_landmark_normalized(lm, self.LEFT_ANKLE)
        
        right_hip = self._get_landmark_normalized(lm, self.RIGHT_HIP)
        right_knee = self._get_landmark_normalized(lm, self.RIGHT_KNEE)
        right_ankle = self._get_landmark_normalized(lm, self.RIGHT_ANKLE)
        
        # Расчет углов коленей (используем нормализованные координаты)
        left_knee_angle = None
        right_knee_angle = None
        
        if all([left_hip, left_knee, left_ankle]):
            left_knee_angle = self._calculate_angle(left_hip, left_knee, left_ankle)
        
        if all([right_hip, right_knee, right_ankle]):
            right_knee_angle = self._calculate_angle(right_hip, right_knee, right_ankle)
        
        # Получаем координаты в пикселях для визуализации
        pose_data = {
            'landmarks': results.pose_landmarks,
            'results': results,
            # Нормализованные координаты (0-1)
            'left_hip_norm': left_hip,
            'left_knee_norm': left_knee,
            'left_ankle_norm': left_ankle,
            'right_hip_norm': right_hip,
            'right_knee_norm': right_knee,
            'right_ankle_norm': right_ankle,
            # Координаты в пикселях
            'left_hip': self._norm_to_pixel(left_hip) if left_hip else None,
            'left_knee': self._norm_to_pixel(left_knee) if left_knee else None,
            'left_ankle': self._norm_to_pixel(left_ankle) if left_ankle else None,
            'right_hip': self._norm_to_pixel(right_hip) if right_hip else None,
            'right_knee': self._norm_to_pixel(right_knee) if right_knee else None,
            'right_ankle': self._norm_to_pixel(right_ankle) if right_ankle else None,
            # Углы
            'left_knee_angle': left_knee_angle,
            'right_knee_angle': right_knee_angle,
            # Все landmarks для UDP
            'all_landmarks': lm
        }
        
        return pose_data
    
    def _get_landmark_normalized(self, landmarks, idx: int) -> Optional[Tuple[float, float]]:
        """
        Получение нормализованных координат ключевой точки (0-1)
        
        Args:
            landmarks: Список landmarks от MediaPipe
            idx: Индекс точки
            
        Returns:
            Кортеж (x, y) в нормализованных координатах или None если точка не видна
        """
        if not landmarks or idx >= len(landmarks):
            return None
        
        landmark = landmarks[idx]
        
        # Проверка видимости точки
        if landmark.visibility < 0.5:
            return None
        
        return (landmark.x, landmark.y)
    
    def _norm_to_pixel(self, norm_coords: Optional[Tuple[float, float]]) -> Optional[Tuple[int, int]]:
        """
        Конвертация нормализованных координат в пиксели
        
        Args:
            norm_coords: Нормализованные координаты (x, y) в диапазоне 0-1
            
        Returns:
            Координаты в пикселях (x, y) или None
        """
        if not norm_coords:
            return None
        
        x = int(norm_coords[0] * self.frame_width)
        y = int(norm_coords[1] * self.frame_height)
        return (x, y)
    
    def _calculate_angle(self, a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
        """
        Расчет угла между тремя точками (как в примере)
        
        Args:
            a: Первая точка (x, y) в нормализованных координатах
            b: Вторая точка (вершина угла) (x, y)
            c: Третья точка (x, y)
            
        Returns:
            Угол в градусах
        """
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)
        
        return np.degrees(angle)
    
    def get_leg_joints(self, pose_data: Optional[Dict]) -> Dict:
        """
        Получение координат суставов ног в формате для UDP
        
        Args:
            pose_data: Данные о позе из process_frame
            
        Returns:
            Словарь с координатами коленей и всех joints
        """
        if not pose_data:
            return {
                'left_knee': None,
                'right_knee': None,
                'left_knee_angle': None,
                'right_knee_angle': None,
                'all_joints': None
            }
        
        # Получаем все joints для UDP (в нормализованных координатах)
        all_joints = None
        if pose_data.get('all_landmarks'):
            lm = pose_data['all_landmarks']
            all_joints = {str(i): [float(l.x), float(l.y), float(getattr(l, 'z', 0.0))] 
                         for i, l in enumerate(lm)}
        
        return {
            'left_knee': pose_data['left_knee'],
            'right_knee': pose_data['right_knee'],
            'left_knee_angle': pose_data['left_knee_angle'],
            'right_knee_angle': pose_data['right_knee_angle'],
            'all_joints': all_joints
        }
    
    def get_skeleton_connections(self):
        """
        Получение списка соединений скелета для визуализации
        
        Returns:
            Список кортежей (start_idx, end_idx)
        """
        return self.skeleton_connections
    
    def release(self):
        """Освобождение ресурсов"""
        if self.pose:
            self.pose.close()

