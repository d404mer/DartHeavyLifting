"""
Модуль отслеживания штанги по окружности (кругляшку сбоку)
"""

import cv2
import numpy as np
import logging
from typing import Optional, List, Tuple
from collections import deque

logger = logging.getLogger(__name__)


class BarbellTracker:
    """Класс для отслеживания штанги по окружности"""
    
    def __init__(self, 
                 min_radius=20, 
                 max_radius=100,
                 param1=50,
                 param2=30,
                 max_path_points=1000):
        """
        Инициализация трекера штанги
        
        Args:
            min_radius: Минимальный радиус окружности штанги
            max_radius: Максимальный радиус окружности штанги
            param1: Параметр для HoughCircles (верхний порог детектора)
            param2: Параметр для HoughCircles (порог аккумулятора)
            max_path_points: Максимальное количество точек пути
        """
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.param1 = param1
        self.param2 = param2
        
        # Хранение пути штанги (x, y, timestamp)
        self.path = deque(maxlen=max_path_points)
        
        # Последнее известное положение
        self.last_position = None
        
        # Счетчик кадров без обнаружения
        self.frames_without_detection = 0
        
    def process_frame(self, frame: np.ndarray, timestamp: float) -> Optional[Tuple[int, int]]:
        """
        Обработка кадра для обнаружения штанги
        
        Args:
            frame: Входной кадр (BGR)
            timestamp: Временная метка кадра
            
        Returns:
            Координаты центра штанги (x, y) или None
        """
        # Конвертация в grayscale для лучшего обнаружения окружностей
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Применение размытия для уменьшения шума
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Дополнительная обработка: применение адаптивной пороговой обработки
        # для лучшего выделения окружностей на разных фонах
        adaptive_thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Обнаружение окружностей с помощью HoughCircles
        # Сначала пробуем на адаптивном пороге, если не получится - на размытом изображении
        circles = cv2.HoughCircles(
            adaptive_thresh,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=self.param1,
            param2=self.param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )
        
        # Если не нашли на адаптивном пороге, пробуем на размытом изображении
        if circles is None:
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=50,
                param1=self.param1,
                param2=self.param2,
                minRadius=self.min_radius,
                maxRadius=self.max_radius
            )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            # Выбираем наиболее вероятную окружность
            # Можно использовать несколько критериев:
            # 1. Если есть предыдущее положение - выбираем ближайшую
            # 2. Иначе выбираем самую большую или центральную
            
            if self.last_position is not None:
                # Выбираем окружность ближайшую к последнему положению
                best_circle = self._find_closest_circle(circles, self.last_position)
            else:
                # Выбираем окружность ближайшую к центру кадра
                h, w = frame.shape[:2]
                center = (w // 2, h // 2)
                best_circle = self._find_closest_circle(circles, center)
            
            if best_circle is not None:
                x, y, r = best_circle
                self.last_position = (x, y)
                self.frames_without_detection = 0
                
                # Добавляем точку в путь
                self.path.append((x, y, timestamp))
                
                logger.debug(f"Штанга обнаружена: ({x}, {y}), радиус: {r}")
                return (x, y)
        
        # Штанга не обнаружена
        self.frames_without_detection += 1
        
        if self.frames_without_detection == 1:
            logger.warning("Штанга не обнаружена на кадре")
        
        return None
    
    def _find_closest_circle(self, circles: np.ndarray, target: Tuple[int, int]) -> Optional[Tuple[int, int, int]]:
        """
        Поиск окружности ближайшей к целевой точке
        
        Args:
            circles: Массив окружностей [(x, y, r), ...]
            target: Целевая точка (x, y)
            
        Returns:
            Окружность (x, y, r) или None
        """
        if len(circles) == 0:
            return None
        
        if len(circles) == 1:
            return tuple(circles[0])
        
        # Вычисляем расстояния до всех окружностей
        distances = []
        for circle in circles:
            x, y, r = circle
            dist = np.sqrt((x - target[0])**2 + (y - target[1])**2)
            distances.append(dist)
        
        # Выбираем ближайшую
        closest_idx = np.argmin(distances)
        return tuple(circles[closest_idx])
    
    def get_path(self) -> List[Tuple[float, float, float]]:
        """
        Получение пути штанги
        
        Returns:
            Список точек пути [(x, y, timestamp), ...]
        """
        return list(self.path)
    
    def clear_path(self):
        """Очистка пути штанги"""
        self.path.clear()
        self.last_position = None
        self.frames_without_detection = 0

