"""
Модуль визуализации позы и пути штанги на видео
Использует MediaPipe для полной визуализации скелета
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Dict, List, Tuple
import config
from scipy.signal import savgol_filter


class Visualizer:
    """Класс для визуализации отслеживания на видео"""
    
    def __init__(self, pose_tracker=None):
        """
        Инициализация визуализатора
        
        Args:
            pose_tracker: Экземпляр PoseTracker для получения соединений скелета
        """
        self.color_barbell = config.COLOR_BARBELL_PATH
        self.color_knee = config.COLOR_KNEE_JOINT
        self.color_leg_bone = config.COLOR_LEG_BONE
        self.color_joint = config.COLOR_JOINT
        self.joint_radius = config.JOINT_RADIUS
        self.line_thickness = config.LINE_THICKNESS
        
        # MediaPipe drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose
        
        # Сохраняем ссылку на pose_tracker для получения соединений
        self.pose_tracker = pose_tracker
    
    def draw_frame(self, 
                   frame: np.ndarray,
                   pose_data: Optional[Dict],
                   barbell_position: Optional[Tuple[int, int]],
                   barbell_path: List[Tuple[float, float, float]]) -> np.ndarray:
        """
        Рисование данных отслеживания на кадре
        
        Args:
            frame: Входной кадр
            pose_data: Данные о позе
            barbell_position: Текущее положение штанги (x, y)
            barbell_path: Путь штанги [(x, y, timestamp), ...]
            
        Returns:
            Кадр с наложенной визуализацией
        """
        result_frame = frame.copy()
        
        # Рисуем путь штанги
        self._draw_barbell_path(result_frame, barbell_path)
        
        # Рисуем текущее положение штанги (отключено)
        # if barbell_position:
        #     self._draw_barbell_position(result_frame, barbell_position)
        
        # Рисуем позу (суставы ног)
        if pose_data:
            self._draw_legs(result_frame, pose_data)
        
        return result_frame
    
    def _draw_barbell_path(self, frame: np.ndarray, path: List[Tuple[float, float, float]]):
        """
        Рисование пути штанги с опциональным сглаживанием
        
        Args:
            frame: Кадр для рисования
            path: Путь штанги [(x, y, timestamp), ...]
        """
        if len(path) < 1:
            return
        
        # Рисуем путь только если есть минимум 2 точки
        if len(path) >= 2:
            xs = np.array([p[0] for p in path], dtype=np.float32)
            ys = np.array([p[1] for p in path], dtype=np.float32)
            
            # Применяем сглаживание если включено
            if config.PATH_SMOOTHING_ENABLED:
                if config.PATH_SMOOTHING_METHOD == "savgol":
                    xs_draw, ys_draw = self._smooth_path_savgol(xs, ys)
                elif config.PATH_SMOOTHING_METHOD == "moving_average":
                    xs_draw, ys_draw = self._smooth_path_moving_average(xs, ys)
                else:
                    xs_draw, ys_draw = xs, ys
            else:
                xs_draw, ys_draw = xs, ys
            
            # Цвет пути - красный (BGR: 0, 0, 255)
            path_color = (0, 0, 255)
            
            # Рисуем линии между соседними точками пути (сглаженные)
            for i in range(1, len(xs_draw)):
                pt1 = (int(xs_draw[i-1]), int(ys_draw[i-1]))
                pt2 = (int(xs_draw[i]), int(ys_draw[i]))
                cv2.line(frame, pt1, pt2, path_color, self.line_thickness)
            
            # Рисуем точки пути (реже, чтобы не перегружать визуализацию)
            step = max(1, len(xs_draw) // 50)  # Рисуем примерно каждую 50-ю точку
            for i in range(0, len(xs_draw), step):
                cv2.circle(frame, (int(xs_draw[i]), int(ys_draw[i])), 2, path_color, -1)
        
        # Рисуем вертикальную пунктирную линию в правой части экрана от первой точки
        if len(path) > 0:
            h, w = frame.shape[:2]
            first_point_y = int(path[0][1])  # Y координата первой точки
            
            # Проверяем, что координата валидна
            if 0 <= first_point_y < h:
                line_x = w - 100  # Позиция линии в правой части экрана (100 пикселей от края для лучшей видимости)
                
                # Рисуем пунктирную линию вверх от первой точки до верха экрана
                dash_length = 15
                gap_length = 8
                current_y = first_point_y
                
                # Используем более толстую линию для видимости
                line_thickness = max(2, self.line_thickness)
                
                # Цвет пунктира - белый (BGR: 255, 255, 255)
                dash_color = (255, 255, 255)
                
                while current_y > 0:
                    # Рисуем сегмент пунктира
                    end_y = max(0, current_y - dash_length)
                    if end_y < current_y:  # Убеждаемся, что есть что рисовать
                        cv2.line(frame, (line_x, current_y), (line_x, end_y), 
                                dash_color, line_thickness)
                    # Переходим к следующему сегменту
                    current_y = end_y - gap_length
                    if current_y <= 0:
                        break
    
    def _smooth_path_savgol(self, xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Сглаживание пути фильтром Савицкого-Голея
        
        Args:
            xs: Массив X координат
            ys: Массив Y координат
            
        Returns:
            Сглаженные массивы (xs, ys)
        """
        if len(xs) < 5:
            return xs, ys
        
        window = int(config.PATH_SAVGOL_WINDOW)
        poly = int(config.PATH_SAVGOL_POLYORDER)
        
        # Окно должно быть нечетным и не больше длины траектории
        if window % 2 == 0:
            window += 1
        if window > len(xs):
            window = len(xs) if len(xs) % 2 == 1 else len(xs) - 1
        if window < (poly + 2):
            window = poly + 3
        if window < 5:
            return xs, ys
        
        # Убеждаемся, что окно нечетное
        if window % 2 == 0:
            window -= 1
        if window < 5:
            return xs, ys
        
        try:
            xs_s = savgol_filter(xs, window_length=window, polyorder=poly, mode='interp')
            ys_s = savgol_filter(ys, window_length=window, polyorder=poly, mode='interp')
            return xs_s, ys_s
        except Exception:
            return xs, ys
    
    def _smooth_path_moving_average(self, xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Сглаживание пути скользящим средним
        
        Args:
            xs: Массив X координат
            ys: Массив Y координат
            
        Returns:
            Сглаженные массивы (xs, ys)
        """
        if len(xs) < 3:
            return xs, ys
        
        window = int(getattr(config, 'PATH_MOVING_AVERAGE_WINDOW', 7))
        
        # Окно должно быть нечетным и не больше длины траектории
        if window % 2 == 0:
            window += 1
        if window > len(xs):
            window = len(xs) if len(xs) % 2 == 1 else len(xs) - 1
        if window < 3:
            return xs, ys
        
        # Применяем скользящее среднее
        # Используем одномерную свертку с равномерным ядром
        kernel = np.ones(window) / window
        
        # Для краев используем padding
        xs_padded = np.pad(xs, (window // 2, window // 2), mode='edge')
        ys_padded = np.pad(ys, (window // 2, window // 2), mode='edge')
        
        xs_s = np.convolve(xs_padded, kernel, mode='valid')
        ys_s = np.convolve(ys_padded, kernel, mode='valid')
        
        return xs_s, ys_s
    
    def _draw_barbell_position(self, frame: np.ndarray, position: Tuple[int, int]):
        """
        Рисование текущего положения штанги
        
        Args:
            frame: Кадр для рисования
            position: Положение штанги (x, y)
        """
        x, y = position
        # Небольшая заполненная точка без креста и большого кольца
        cv2.circle(frame, (int(x), int(y)), 6, self.color_barbell, -1)
    
    def _draw_legs(self, frame: np.ndarray, pose_data: Dict):
        """
        Рисование полного скелета через MediaPipe
        
        Args:
            frame: Кадр для рисования
            pose_data: Данные о позе
        """
        if not pose_data.get('landmarks'):
            return
        
        landmarks = pose_data['landmarks']
        h, w = frame.shape[:2]
        
        # Рисуем скелет используя соединения из pose_tracker
        if self.pose_tracker:
            skeleton_connections = self.pose_tracker.get_skeleton_connections()
            
            # Конвертируем landmarks в пиксели
            points_px = []
            for landmark in landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                points_px.append((x, y))
            
            # Рисуем линии скелета
            for start_idx, end_idx in skeleton_connections:
                if start_idx < len(points_px) and end_idx < len(points_px):
                    pt1 = points_px[start_idx]
                    pt2 = points_px[end_idx]
                    # Проверяем видимость точек
                    if (landmarks.landmark[start_idx].visibility > 0.5 and 
                        landmarks.landmark[end_idx].visibility > 0.5):
                        cv2.line(frame, pt1, pt2, self.color_leg_bone, self.line_thickness)
            
            # Рисуем точки суставов (торс, ноги, руки основные)
            display_points = [
                # Торс
                11, 12, 23, 24,
                # Левая нога
                25, 27, 29, 31,
                # Правая нога
                26, 28, 30, 32,
                # Руки основные
                13, 15, 14, 16
            ]
            
            for idx in display_points:
                if idx < len(points_px) and landmarks.landmark[idx].visibility > 0.5:
                    point = points_px[idx]
                    # Колени рисуем другим цветом
                    if idx in [25, 26]:  # LEFT_KNEE, RIGHT_KNEE
                        cv2.circle(frame, point, self.joint_radius + 2, self.color_knee, -1)
                        
                        # Отображаем угол колена
                        if idx == 25 and pose_data.get('left_knee_angle'):
                            angle = pose_data['left_knee_angle']
                            angle_text = f"{int(angle)}°"
                            text_pos = (point[0] + 15, point[1] - 15)
                            cv2.putText(frame, angle_text, text_pos,
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color_knee, 2)
                        elif idx == 26 and pose_data.get('right_knee_angle'):
                            angle = pose_data['right_knee_angle']
                            angle_text = f"{int(angle)}°"
                            text_pos = (point[0] + 15, point[1] - 15)
                            cv2.putText(frame, angle_text, text_pos,
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color_knee, 2)
                    else:
                        cv2.circle(frame, point, self.joint_radius, self.color_joint, -1)
    
    def _draw_line(self, frame: np.ndarray, pt1: Tuple[float, float], pt2: Tuple[float, float], color: Tuple[int, int, int]):
        """
        Рисование линии между двумя точками
        
        Args:
            frame: Кадр для рисования
            pt1: Первая точка
            pt2: Вторая точка
            color: Цвет линии (BGR)
        """
        cv2.line(frame, 
                (int(pt1[0]), int(pt1[1])),
                (int(pt2[0]), int(pt2[1])),
                color, self.line_thickness)
    
    def _draw_joint(self, frame: np.ndarray, position: Tuple[float, float], color: Tuple[int, int, int]):
        """
        Рисование сустава (точка)
        
        Args:
            frame: Кадр для рисования
            position: Положение сустава
            color: Цвет точки (BGR)
        """
        cv2.circle(frame, (int(position[0]), int(position[1])), self.joint_radius, color, -1)

