"""
Объединенная версия: GUI + трекинг позы + трекинг штанги
Объединяет функциональность main_pose_tracker.py и main.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
import threading
import queue
import time
import socket
import json
import gc
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.colorchooser import askcolor
from typing import Optional, Tuple, List
from collections import deque

# Импорты из проекта
import config
from pose_tracker import PoseTracker
from visualizer import Visualizer

# Попытка NDI
try:
    import NDIlib as ndi
    NDI_AVAILABLE = True
except Exception:
    NDI_AVAILABLE = False

# Попытка виртуальной камеры
try:
    import pyvirtualcam
    from pyvirtualcam import PixelFormat
    VIRTUALCAM_AVAILABLE = True
except Exception:
    VIRTUALCAM_AVAILABLE = False

# MediaPipe
import mediapipe as mp
mp_pose = mp.solutions.pose

# UDP
try:
    UE_IP, UE_PORT = config.UDP_HOST, config.UDP_PORT
except:
    UE_IP, UE_PORT = "127.0.0.1", 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# -------------------- Утилиты --------------------
def list_cameras(max_test=6):
    """Список доступных камер"""
    cams = []
    for i in range(max_test):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if os.name == "nt" else 0)
        if cap and cap.isOpened():
            ret, _ = cap.read()
            if ret:
                cams.append(i)
            cap.release()
    return cams

def calculate_angle(a, b, c):
    """Расчет угла между тремя точками"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return 0.0
    cosang = np.dot(ba, bc) / denom
    return float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))

def resize_with_aspect(frame, target_w, target_h):
    """Изменение размера с сохранением пропорций"""
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w*scale), int(h*scale)
    resized = cv2.resize(frame, (new_w, new_h))
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x = (target_w - new_w)//2
    y = (target_h - new_h)//2
    canvas[y:y+new_h, x:x+new_w] = resized
    return canvas

# -------------------- OptimizedBarbellTracker (из main.py) --------------------
class OptimizedBarbellTracker:
    """Оптимизированный трекер штанги"""
    
    def __init__(self, smoothing_factor=0.0):
        self.path = deque(maxlen=config.MAX_PATH_POINTS)
        self.last_position = None
        self.smoothed_position = None
        self.frames_without_detection = 0
        self.search_region = None
        self.smoothing_factor = smoothing_factor
        self.last_radius = None
        self.last_confidence = None
        self.last_detection_source = None
        self._kalman = None
        self._jitter_buffer = deque(maxlen=max(3, int(getattr(config, 'BARBELL_ANTI_JITTER_WINDOW', 3))))
        self._last_motion_ts = None
        self._last_motion_pos = None
    
    class _Kalman2D:
        """Калман фильтр для 2D позиции"""
        def __init__(self, x, y):
            self.dt = 1 / max(1, config.TARGET_FPS)
            self.x = np.array([[x], [y], [0.0], [0.0]], dtype=np.float32)
            self.F = np.array([[1, 0, self.dt, 0], [0, 1, 0, self.dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
            self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
            q, r = 2.0, 25.0
            self.Q = np.eye(4, dtype=np.float32) * q
            self.R = np.eye(2, dtype=np.float32) * r
            self.P = np.eye(4, dtype=np.float32) * 100.0
        
        def predict(self):
            self.x = self.F @ self.x
            self.P = self.F @ self.P @ self.F.T + self.Q
            return float(self.x[0, 0]), float(self.x[1, 0])
        
        def update(self, zx, zy):
            z = np.array([[zx], [zy]], dtype=np.float32)
            y = z - (self.H @ self.x)
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)
            self.x = self.x + K @ y
            I = np.eye(self.P.shape[0], dtype=np.float32)
            self.P = (I - K @ self.H) @ self.P
            return float(self.x[0, 0]), float(self.x[1, 0])
    
    def update_search_region(self, left_wrist, right_wrist, frame_shape):
        """Обновление области поиска на основе положения рук"""
        if left_wrist and right_wrist:
            min_x = min(left_wrist[0], right_wrist[0]) - 100
            max_x = max(left_wrist[0], right_wrist[0]) + 100
            min_y = min(left_wrist[1], right_wrist[1]) - 150
            max_y = max(left_wrist[1], right_wrist[1]) + 50
            h, w = frame_shape[:2]
            min_x, max_x = max(0, min_x), min(w, max_x)
            min_y, max_y = max(0, min_y), min(h, max_y)
            self.search_region = (int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y))
        else:
            self.search_region = None
    
    def detect_barbell(self, frame: np.ndarray, timestamp: float, debug_frame=None) -> Optional[Tuple[int, int]]:
        """Обнаружение штанги"""
        # Используем область поиска если доступна
        if config.BARBELL_USE_SEARCH_REGION and self.search_region:
            x, y, w, h = self.search_region
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0:
                self.search_region = None
                return None
            if config.BARBELL_DEBUG_MODE and debug_frame is not None:
                cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
        else:
            roi = frame
            x, y = 0, 0
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        if config.BARBELL_ENABLE_CLAHE:
            clahe = cv2.createCLAHE(clipLimit=config.BARBELL_CLAHE_CLIP_LIMIT, tileGridSize=config.BARBELL_CLAHE_TILE_GRID_SIZE)
            gray = clahe.apply(gray)
        
        median = cv2.medianBlur(gray, 5)
        blurred = cv2.GaussianBlur(median, (config.BARBELL_BLUR_SIZE, config.BARBELL_BLUR_SIZE), config.BARBELL_BLUR_SIGMA)
        
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT,
            dp=config.BARBELL_CIRCLE_DP,
            minDist=config.BARBELL_CIRCLE_MIN_DIST,
            param1=config.BARBELL_CIRCLE_PARAM1,
            param2=config.BARBELL_CIRCLE_PARAM2,
            minRadius=config.BARBELL_CIRCLE_MIN_RADIUS,
            maxRadius=config.BARBELL_CIRCLE_MAX_RADIUS
        )
        detection_source = 'hough'
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            best_circle = self._select_best_circle(circles)
            
            if best_circle is not None:
                cx, cy, r = best_circle
                global_x = cx + x
                global_y = cy + y
                
                if not config.BARBELL_USE_KALMAN:
                    px, py = float(global_x), float(global_y)
                    if getattr(config, 'BARBELL_ANTI_JITTER_2TAP', True) and (self.smoothed_position or self.last_position):
                        prevx, prevy = (self.smoothed_position or self.last_position)
                        speed = np.hypot(px - prevx, py - prevy)
                        low = getattr(config, 'BARBELL_ANTI_JITTER_SPEED_THRESH_LOW', 2.0)
                        high = getattr(config, 'BARBELL_ANTI_JITTER_SPEED_THRESH_HIGH', 6.0)
                        w = float(getattr(config, 'BARBELL_ANTI_JITTER_2TAP_WEIGHT', 0.6))
                        if speed <= low:
                            px, py = float(prevx), float(prevy)
                        elif speed <= high:
                            px = w * px + (1 - w) * prevx
                            py = w * py + (1 - w) * prevy
                    self.smoothed_position = (px, py)
                else:
                    if self._kalman is None:
                        self._kalman = self._Kalman2D(global_x, global_y)
                    kx, ky = self._kalman.update(global_x, global_y)
                    if self.smoothed_position is None:
                        self.smoothed_position = (float(kx), float(ky))
                    else:
                        smooth_x = self.smoothing_factor * self.smoothed_position[0] + (1 - self.smoothing_factor) * kx
                        smooth_y = self.smoothing_factor * self.smoothed_position[1] + (1 - self.smoothing_factor) * ky
                        self.smoothed_position = (smooth_x, smooth_y)
                
                self.last_position = (global_x, global_y)
                self.frames_without_detection = 0
                self.last_radius = r
                self.last_confidence = 0.9
                self.last_detection_source = detection_source
                self.path.append((self.smoothed_position[0], self.smoothed_position[1], timestamp))
                return (int(self.smoothed_position[0]), int(self.smoothed_position[1]))
        
        self.frames_without_detection += 1
        if self.smoothed_position is not None and self.frames_without_detection < 5:
            if len(self.path) >= 2:
                prev_x, prev_y, _ = self.path[-1]
                return (int(prev_x), int(prev_y))
            return (int(self.smoothed_position[0]), int(self.smoothed_position[1]))
        return None
    
    def _select_best_circle(self, circles: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """Выбор лучшей окружности"""
        if len(circles) == 0:
            return None
        if len(circles) == 1:
            return tuple(circles[0])
        if self.last_position is not None:
            best_circle = None
            best_score = -1
            last_x, last_y = self.last_position
            for circle in circles:
                cx, cy, r = circle
                dx, dy = float(cx - last_x), float(cy - last_y)
                gx = float(getattr(config, 'BARBELL_X_STABILITY_GAIN', 2.0))
                position_dist = np.sqrt((gx * dx)**2 + (dy)**2)
                position_score = 1.0 / (1.0 + position_dist / 100.0)
                if position_score > best_score:
                    best_score = position_score
                    best_circle = circle
            if best_circle is not None:
                return tuple(best_circle)
        if config.BARBELL_PREFER_LARGER_RADIUS:
            largest_idx = np.argmax([c[2] for c in circles])
            return tuple(circles[largest_idx])
        return tuple(circles[0])
    
    def get_path(self) -> List[Tuple[float, float, float]]:
        return list(self.path)
    
    def clear_path(self):
        self.path.clear()
        self.last_position = None
        self.smoothed_position = None
        self.frames_without_detection = 0

# -------------------- Потоки обработки --------------------
class CaptureThread(threading.Thread):
    """Поток захвата видео"""
    def __init__(self, source, out_q, stop_event, target_fps=30):
        super().__init__(daemon=True)
        self.source = source
        self.out_q = out_q
        self.stop_event = stop_event
        self.target_fps = target_fps
        self.cap = None
        self.is_video_file = False
        self.video_fps = 30.0
        self.open_source(source)
    
    def open_source(self, source):
        if isinstance(source, str) and source.lower().endswith((".mp4", ".mov", ".avi")):
            self.cap = cv2.VideoCapture(source)
            self.is_video_file = True
            # Получаем FPS видео файла
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                self.video_fps = fps
            else:
                self.video_fps = self.target_fps
            print(f"📹 Видео файл открыт, FPS: {self.video_fps:.2f}")
        else:
            try:
                idx = int(source)
                self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW if os.name == "nt" else 0)
                self.is_video_file = False
            except:
                self.cap = cv2.VideoCapture(source)
                self.is_video_file = False
        if not self.cap.isOpened():
            print("❌ Cannot open source:", source)
    
    def run(self):
        frame_time = 1.0 / self.video_fps if self.is_video_file else 1.0 / self.target_fps
        last_frame_time = time.time()
        
        while not self.stop_event.is_set():
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.05)
                continue
            
            # Контроль скорости для видео файлов
            if self.is_video_file:
                elapsed = time.time() - last_frame_time
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)
                last_frame_time = time.time()
            
            ret, frame = self.cap.read()
            if not ret:
                if self.is_video_file:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    last_frame_time = time.time()
                    continue
                time.sleep(0.02)
                continue
            try:
                self.out_q.put(frame, block=False)
            except queue.Full:
                try:
                    _ = self.out_q.get_nowait()
                    self.out_q.put(frame, block=False)
                except:
                    pass
        try:
            self.cap.release()
        except:
            pass

class ProcThread(threading.Thread):
    """Поток обработки (MediaPipe + трекинг штанги)"""
    def __init__(self, in_q, out_q, stop_event, proc_w, proc_h, every_n, pose_tracker, barbell_tracker, enable_barbell):
        super().__init__(daemon=True)
        self.in_q = in_q
        self.out_q = out_q
        self.stop_event = stop_event
        self.proc_w = proc_w
        self.proc_h = proc_h
        self.every_n = every_n
        self.idx = 0
        self.pose_tracker = pose_tracker
        self.barbell_tracker = barbell_tracker
        self.enable_barbell = enable_barbell
    
    def run(self):
        while not self.stop_event.is_set():
            try:
                frame = self.in_q.get(timeout=0.05)
            except queue.Empty:
                time.sleep(0.01)
                continue
            
            self.idx += 1
            timestamp = time.time()
            pose_data = None
            barbell_pos = None
            
            # Обработка позы
            if self.pose_tracker and self.idx % self.every_n == 0:
                pose_data = self.pose_tracker.process_frame(frame)
                
                # Обновление области поиска штанги
                if self.enable_barbell and pose_data and config.BARBELL_USE_SEARCH_REGION and pose_data.get('all_landmarks'):
                    lm = pose_data['all_landmarks']
                    h, w = frame.shape[:2]
                    LEFT_WRIST, RIGHT_WRIST = 15, 16
                    left_wrist_px = None
                    right_wrist_px = None
                    if lm[LEFT_WRIST].visibility > 0.5:
                        left_wrist_px = (int(lm[LEFT_WRIST].x * w), int(lm[LEFT_WRIST].y * h))
                    if lm[RIGHT_WRIST].visibility > 0.5:
                        right_wrist_px = (int(lm[RIGHT_WRIST].x * w), int(lm[RIGHT_WRIST].y * h))
                    if left_wrist_px or right_wrist_px:
                        self.barbell_tracker.update_search_region(left_wrist_px, right_wrist_px, frame.shape)
            
            # Обработка штанги
            if self.enable_barbell:
                barbell_pos = self.barbell_tracker.detect_barbell(frame, timestamp)
            
            try:
                self.out_q.put((frame, pose_data, barbell_pos, timestamp), block=False)
            except queue.Full:
                try:
                    _ = self.out_q.get_nowait()
                    self.out_q.put((frame, pose_data, barbell_pos, timestamp), block=False)
                except:
                    pass

# -------------------- Отрисовка скелета --------------------
def draw_overlay(frame, landmarks, angles, bone_color, joint_color, bone_width, joint_radius):
    """Рисует скелет с углами"""
    if landmarks is None:
        return frame
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    limbs = {
        "left_arm": (11, 13, 15),
        "right_arm": (12, 14, 16),
        "left_leg": (23, 25, 27),
        "right_leg": (24, 26, 28),
    }
    
    bone_bgr = tuple(int(bone_color[i:i+2], 16) for i in (5, 3, 1))
    joint_bgr = tuple(int(joint_color[i:i+2], 16) for i in (5, 3, 1))
    
    for limb, (a, b, c) in limbs.items():
        try:
            pa = (int(landmarks[a].x * w), int(landmarks[a].y * h))
            pb = (int(landmarks[b].x * w), int(landmarks[b].y * h))
            pc = (int(landmarks[c].x * w), int(landmarks[c].y * h))
        except:
            continue
        
        outline_width = bone_width + 2
        cv2.line(overlay, pa, pb, (0, 0, 0), outline_width, cv2.LINE_AA)
        cv2.line(overlay, pb, pc, (0, 0, 0), outline_width, cv2.LINE_AA)
        cv2.line(overlay, pa, pb, bone_bgr, bone_width, cv2.LINE_AA)
        cv2.line(overlay, pb, pc, bone_bgr, bone_width, cv2.LINE_AA)
        
        angle_val = angles.get(limb, 0.0)
        cv2.putText(overlay, f"{angle_val:.0f}°", (pb[0] + 10, pb[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(overlay, f"{angle_val:.0f}°", (pb[0] + 10, pb[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, bone_bgr, 1, cv2.LINE_AA)
        
        for idx in [a, b, c]:
            x, y = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
            cv2.circle(overlay, (x, y), joint_radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(overlay, (x, y), joint_radius, joint_bgr, -1, cv2.LINE_AA)
    
    return cv2.addWeighted(overlay, 0.9, frame, 0.1, 0)

# -------------------- GUI --------------------
class App:
    def __init__(self, root):
        self.root = root
        root.title("Unified Pose & Barbell Tracking")
        root.configure(bg='#2b2b2b')
        self.running = False
        self.stop_event = threading.Event()
        
        # Стили
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#2b2b2b')
        self.style.configure('TLabel', background='#2b2b2b', foreground='white')
        self.style.configure('TLabelframe', background='#2b2b2b', foreground='white')
        self.style.configure('TLabelframe.Label', background='#2b2b2b', foreground='white')
        self.style.configure('TButton', background='#404040', foreground='white')
        self.style.configure('TCheckbutton', background='#2b2b2b', foreground='white')
        self.style.configure('TCombobox', background='#404040', foreground='white')
        self.style.configure('TEntry', background='#404040', foreground='white')
        self.style.configure('TScale', background='#2b2b2b')
        self.style.configure('TRadiobutton', background='#2b2b2b', foreground='white')
        
        # Параметры
        self.WINDOW_W = 1920
        self.WINDOW_H = 1080
        self.proc_w = 320
        self.proc_h = 180
        self.every_n = 2
        self.target_fps = 30
        
        # Режим работы: "pose", "barbell", "both"
        self.mode = tk.StringVar(value="both")
        self.enable_pose = tk.BooleanVar(value=True)
        self.enable_barbell = tk.BooleanVar(value=True)
        
        # GUI элементы
        self.ndi_name = tk.StringVar(value="UnifiedStream_NDI")
        self.use_ndi = tk.BooleanVar(value=False and NDI_AVAILABLE)
        self.use_virtual = tk.BooleanVar(value=False and VIRTUALCAM_AVAILABLE)
        self.show_joints = tk.BooleanVar(value=True)
        self.model_complexity = tk.IntVar(value=1)
        self.smooth_landmarks = tk.BooleanVar(value=True)
        self.min_det = tk.DoubleVar(value=0.4)
        self.min_track = tk.DoubleVar(value=0.4)
        self.bone_color = tk.StringVar(value="#FF6B35")
        self.joint_color = tk.StringVar(value="#4ECDC4")
        self.bone_width = tk.IntVar(value=6)
        self.joint_radius = tk.IntVar(value=6)
        
        # Основной layout
        main_frame = ttk.Frame(root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        left_frame = ttk.Frame(main_frame, width=400)
        left_frame.pack(side='left', fill='y', padx=(0, 10))
        left_frame.pack_propagate(False)
        
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side='right', fill='both', expand=True)
        
        # Предпросмотр
        preview_header = ttk.Label(right_frame, text="Предпросмотр", font=('Arial', 12, 'bold'))
        preview_header.pack(pady=(0, 5))
        preview_container = ttk.Frame(right_frame)
        preview_container.pack(fill='both', expand=True)
        self.preview_label = ttk.Label(preview_container, text="Запустите стрим для предпросмотра",
                                      background='black', foreground='white', font=('Arial', 10), anchor='center')
        self.preview_label.pack(fill='both', expand=True)
        
        # === РЕЖИМ РАБОТЫ ===
        mode_frame = ttk.LabelFrame(left_frame, text="🎯 Режим работы", padding=10)
        mode_frame.pack(fill='x', pady=(0, 10))
        ttk.Radiobutton(mode_frame, text="Только поза", variable=self.mode, value="pose",
                       command=self.on_mode_change).pack(anchor='w', pady=2)
        ttk.Radiobutton(mode_frame, text="Только штанга", variable=self.mode, value="barbell",
                       command=self.on_mode_change).pack(anchor='w', pady=2)
        ttk.Radiobutton(mode_frame, text="Поза + штанга", variable=self.mode, value="both",
                       command=self.on_mode_change).pack(anchor='w', pady=2)
        
        # === ИСТОЧНИК ВИДЕО ===
        source_frame = ttk.LabelFrame(left_frame, text="📷 Источник видео", padding=10)
        source_frame.pack(fill='x', pady=(0, 10))
        
        source_row1 = ttk.Frame(source_frame)
        source_row1.pack(fill='x', pady=2)
        ttk.Label(source_row1, text="Камера:").pack(side='left')
        self.cam_list = list_cameras(6)
        self.source_var = tk.StringVar(value=str(self.cam_list[0]) if self.cam_list else "0")
        self.source_combo = ttk.Combobox(source_row1, values=[str(x) for x in self.cam_list],
                                       textvariable=self.source_var, width=12)
        self.source_combo.pack(side='left', padx=5)
        
        source_row2 = ttk.Frame(source_frame)
        source_row2.pack(fill='x', pady=2)
        ttk.Button(source_row2, text="📁 Выбрать видео", command=self.browse_file).pack(side='left', padx=2)
        ttk.Button(source_row2, text="🔄 Обновить камеры", command=self.refresh_cams).pack(side='left', padx=2)
        
        # Поле для отображения выбранного файла
        source_row3 = ttk.Frame(source_frame)
        source_row3.pack(fill='x', pady=2)
        ttk.Label(source_row3, text="Файл:").pack(side='left')
        self.file_label = ttk.Label(source_row3, text="(не выбран)", foreground='gray', font=('Arial', 8))
        self.file_label.pack(side='left', padx=5)
        
        # === НАСТРОЙКИ ОБРАБОТКИ ===
        processing_frame = ttk.LabelFrame(left_frame, text="⚙️ Настройки обработки", padding=10)
        processing_frame.pack(fill='x', pady=(0, 10))
        proc_row1 = ttk.Frame(processing_frame)
        proc_row1.pack(fill='x', pady=2)
        ttk.Label(proc_row1, text="Разрешение:").pack(side='left')
        self.proc_entry = ttk.Entry(proc_row1, width=10)
        self.proc_entry.insert(0, f"{self.proc_w}x{self.proc_h}")
        self.proc_entry.pack(side='left', padx=5)
        ttk.Label(proc_row1, text="Кадры:").pack(side='left', padx=(10,0))
        self.every_spin = ttk.Spinbox(proc_row1, from_=1, to=6, width=4)
        self.every_spin.delete(0, "end")
        self.every_spin.insert(0, str(self.every_n))
        self.every_spin.pack(side='left', padx=5)
        proc_row2 = ttk.Frame(processing_frame)
        proc_row2.pack(fill='x', pady=2)
        ttk.Label(proc_row2, text="FPS:").pack(side='left')
        self.fps_spin = ttk.Spinbox(proc_row2, from_=5, to=60, width=4)
        self.fps_spin.delete(0, "end")
        self.fps_spin.insert(0, str(self.target_fps))
        self.fps_spin.pack(side='left', padx=5)
        
        # === ВНЕШНИЙ ВИД ===
        appearance_frame = ttk.LabelFrame(left_frame, text="🎨 Внешний вид", padding=10)
        appearance_frame.pack(fill='x', pady=(0, 10))
        colors_frame = ttk.Frame(appearance_frame)
        colors_frame.pack(fill='x', pady=5)
        ttk.Label(colors_frame, text="Цвет костей:").grid(row=0, column=0, sticky="w", padx=5, pady=3)
        self.bone_color_btn = ttk.Button(colors_frame, text="Выбрать", command=self.choose_bone_color, width=8)
        self.bone_color_btn.grid(row=0, column=1, padx=5, pady=3)
        self.bone_color_preview = tk.Canvas(colors_frame, width=40, height=20, bg=self.bone_color.get(), relief='solid', bd=1)
        self.bone_color_preview.grid(row=0, column=2, padx=5, pady=3)
        ttk.Label(colors_frame, text="Цвет суставов:").grid(row=1, column=0, sticky="w", padx=5, pady=3)
        self.joint_color_btn = ttk.Button(colors_frame, text="Выбрать", command=self.choose_joint_color, width=8)
        self.joint_color_btn.grid(row=1, column=1, padx=5, pady=3)
        self.joint_color_preview = tk.Canvas(colors_frame, width=40, height=20, bg=self.joint_color.get(), relief='solid', bd=1)
        self.joint_color_preview.grid(row=1, column=2, padx=5, pady=3)
        sizes_frame = ttk.Frame(appearance_frame)
        sizes_frame.pack(fill='x', pady=5)
        ttk.Label(sizes_frame, text="Толщина костей:").grid(row=0, column=0, sticky="w", padx=5, pady=3)
        bone_scale_frame = ttk.Frame(sizes_frame)
        bone_scale_frame.grid(row=0, column=1, columnspan=2, sticky='ew', padx=5, pady=3)
        ttk.Scale(bone_scale_frame, from_=1, to=20, orient='horizontal', variable=self.bone_width,
                 command=self.on_bone_width_change, length=120).pack(side='left')
        self.bone_width_label = ttk.Label(bone_scale_frame, text=str(self.bone_width.get()), width=3)
        self.bone_width_label.pack(side='left', padx=5)
        ttk.Label(sizes_frame, text="Размер суставов:").grid(row=1, column=0, sticky="w", padx=5, pady=3)
        joint_scale_frame = ttk.Frame(sizes_frame)
        joint_scale_frame.grid(row=1, column=1, columnspan=2, sticky='ew', padx=5, pady=3)
        ttk.Scale(joint_scale_frame, from_=1, to=20, orient='horizontal', variable=self.joint_radius,
                 command=self.on_joint_radius_change, length=120).pack(side='left')
        self.joint_radius_label = ttk.Label(joint_scale_frame, text=str(self.joint_radius.get()), width=3)
        self.joint_radius_label.pack(side='left', padx=5)
        
        # === МОДЕЛЬ ===
        model_frame = ttk.LabelFrame(left_frame, text="🧠 Настройки модели", padding=10)
        model_frame.pack(fill='x', pady=(0, 10))
        model_row1 = ttk.Frame(model_frame)
        model_row1.pack(fill='x', pady=2)
        ttk.Label(model_row1, text="Сложность:").pack(side='left')
        ttk.Spinbox(model_row1, from_=0, to=1, width=5, textvariable=self.model_complexity).pack(side='left', padx=5)
        ttk.Checkbutton(model_row1, text="Сглаживание", variable=self.smooth_landmarks).pack(side='left', padx=10)
        model_row2 = ttk.Frame(model_frame)
        model_row2.pack(fill='x', pady=2)
        ttk.Label(model_row2, text="Детекция:").pack(side='left')
        ttk.Entry(model_row2, textvariable=self.min_det, width=6).pack(side='left', padx=5)
        ttk.Label(model_row2, text="Трекинг:").pack(side='left', padx=(10,0))
        ttk.Entry(model_row2, textvariable=self.min_track, width=6).pack(side='left', padx=5)
        
        # === ВЫХОДНЫЕ ПОТОКИ ===
        output_frame = ttk.LabelFrame(left_frame, text="📤 Выходные потоки", padding=10)
        output_frame.pack(fill='x', pady=(0, 10))
        ttk.Checkbutton(output_frame, text="Показывать скелет", variable=self.show_joints).pack(anchor='w', pady=2)
        ttk.Checkbutton(output_frame, text="Использовать NDI", variable=self.use_ndi).pack(anchor='w', pady=2)
        ttk.Checkbutton(output_frame, text="Виртуальная камера", variable=self.use_virtual).pack(anchor='w', pady=2)
        ndi_frame = ttk.Frame(output_frame)
        ndi_frame.pack(fill='x', pady=2)
        ttk.Label(ndi_frame, text="Имя NDI:").pack(side='left')
        ttk.Entry(ndi_frame, textvariable=self.ndi_name, width=15).pack(side='left', padx=5)
        
        # === УПРАВЛЕНИЕ ===
        control_frame = ttk.Frame(left_frame)
        control_frame.pack(fill='x', pady=10)
        self.start_btn = ttk.Button(control_frame, text="▶️ Запуск", command=self.start, width=12)
        self.start_btn.pack(side='left', padx=2)
        self.stop_btn = ttk.Button(control_frame, text="⏹️ Остановка", command=self.stop, state="disabled", width=12)
        self.stop_btn.pack(side='left', padx=2)
        ttk.Button(control_frame, text="❌ Выход", command=self.quit, width=12).pack(side='left', padx=2)
        
        # === СТАТУС ===
        status_frame = ttk.Frame(left_frame)
        status_frame.pack(fill='x', pady=5)
        self.status_var = tk.StringVar(value="Готов к работе")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, relief="sunken",
                               anchor="center", background='#404040', foreground='white')
        status_label.pack(fill='x')
        
        # Внутренние переменные
        self.cap_thread = None
        self.proc_thread = None
        self.proc_q = None
        self.render_q = None
        self.ndi_sender = None
        self.virtual_cam = None
        self.pose_tracker = None
        self.barbell_tracker = None
        self.visualizer = None
        self.angle_buffer = {k: [] for k in ["left_arm","right_arm","left_leg","right_leg"]}
        self.frame_counter = 0
    
    def on_mode_change(self):
        """Обновление режима работы"""
        mode = self.mode.get()
        self.enable_pose.set(mode == "pose" or mode == "both")
        self.enable_barbell.set(mode == "barbell" or mode == "both")
    
    def choose_bone_color(self):
        color = askcolor(initialcolor=self.bone_color.get(), title="Выберите цвет костей")[1]
        if color:
            self.bone_color.set(color)
            self.bone_color_preview.config(bg=color)
    
    def choose_joint_color(self):
        color = askcolor(initialcolor=self.joint_color.get(), title="Выберите цвет суставов")[1]
        if color:
            self.joint_color.set(color)
            self.joint_color_preview.config(bg=color)
    
    def on_bone_width_change(self, value):
        self.bone_width_label.config(text=str(int(float(value))))
    
    def on_joint_radius_change(self, value):
        self.joint_radius_label.config(text=str(int(float(value))))
    
    def browse_file(self):
        path = filedialog.askopenfilename(
            title="Выберите видео файл",
            filetypes=[
                ("Video files", "*.mp4 *.mov *.avi *.MP4 *.MOV *.AVI"),
                ("MP4 files", "*.mp4 *.MP4"),
                ("MOV files", "*.mov *.MOV"),
                ("AVI files", "*.avi *.AVI"),
                ("All files", "*.*")
            ],
            initialdir="vids" if os.path.exists("vids") else "."
        )
        if path:
            self.source_var.set(path)
            # Показываем имя файла (только имя, не полный путь)
            filename = os.path.basename(path)
            if len(filename) > 30:
                filename = "..." + filename[-27:]
            self.file_label.config(text=filename, foreground='white')
            # Обновляем список в ComboBox, чтобы путь к файлу был доступен
            current_values = list(self.source_combo['values'])
            if path not in current_values:
                self.source_combo['values'] = current_values + [path]
    
    def refresh_cams(self):
        cams = list_cameras(8)
        # Сохраняем текущее значение (может быть путь к файлу)
        current_value = self.source_var.get()
        # Обновляем список камер
        cam_values = [str(x) for x in cams]
        # Если текущее значение - это путь к файлу, добавляем его в список
        if current_value and (current_value.lower().endswith((".mp4", ".mov", ".avi")) or os.path.exists(current_value)):
            if current_value not in cam_values:
                cam_values.append(current_value)
        self.source_combo['values'] = cam_values
        # Если текущее значение - это камера, обновляем список, иначе оставляем как есть
        if current_value and current_value.isdigit() and int(current_value) in cams:
            pass  # Оставляем текущую камеру
        elif cams and not (current_value and os.path.exists(current_value)):
            self.source_var.set(str(cams[0]))
    
    def start(self):
        if self.running:
            return
        self.on_mode_change()
        src = self.source_var.get()
        proc_res = self.proc_entry.get().strip()
        try:
            pw, ph = [int(x) for x in proc_res.split("x")]
        except Exception:
            messagebox.showerror("Error", "Разрешение должно быть в формате 320x180")
            return
        self.proc_w, self.proc_h = pw, ph
        try:
            self.every_n = max(1, int(self.every_spin.get()))
        except:
            self.every_n = 2
        try:
            self.target_fps = int(self.fps_spin.get())
        except:
            self.target_fps = 30
        
        # Инициализация трекеров
        self.status_var.set("Инициализация...")
        if self.enable_pose.get():
            self.pose_tracker = PoseTracker(
                min_detection_confidence=self.min_det.get(),
                min_tracking_confidence=self.min_track.get()
            )
        if self.enable_barbell.get():
            self.barbell_tracker = OptimizedBarbellTracker(smoothing_factor=config.BARBELL_SMOOTHING_FACTOR)
        # Visualizer нужен для отрисовки пути штанги
        if self.enable_barbell.get() or self.enable_pose.get():
            self.visualizer = Visualizer(pose_tracker=self.pose_tracker if self.enable_pose.get() else None)
        
        # Очереди и потоки
        self.proc_q = queue.Queue(maxsize=2)
        self.render_q = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()
        
        self.cap_thread = CaptureThread(src, self.proc_q, self.stop_event, self.target_fps)
        self.proc_thread = ProcThread(self.proc_q, self.render_q, self.stop_event, self.proc_w, self.proc_h,
                                     self.every_n, self.pose_tracker, self.barbell_tracker, self.enable_barbell.get())
        self.cap_thread.start()
        self.proc_thread.start()
        
        # NDI и виртуальная камера
        if self.use_ndi.get() and NDI_AVAILABLE:
            try:
                if ndi.initialize():
                    sc = ndi.SendCreate()
                    sc.ndi_name = self.ndi_name.get()
                    self.ndi_sender = ndi.send_create(sc)
            except Exception as e:
                messagebox.showwarning("NDI", f"NDI init error: {e}")
        if self.use_virtual.get() and VIRTUALCAM_AVAILABLE:
            try:
                self.virtual_cam = pyvirtualcam.Camera(width=self.WINDOW_W, height=self.WINDOW_H,
                                                      fps=self.target_fps, fmt=PixelFormat.BGR)
            except Exception as e:
                messagebox.showwarning("VirtualCam", f"Error: {e}")
        
        self.running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_var.set("Стриминг активен")
        
        # Выводим информацию о горячих клавишах
        print("\n" + "="*50)
        print("🎮 ГОРЯЧИЕ КЛАВИШИ (в окне OpenCV):")
        print("  'q' или ESC - остановка")
        print("  'c' - очистить путь штанги")
        print("  '1' - переключить трекинг позы")
        print("  '2' - переключить трекинг штанги")
        print("="*50 + "\n")
        
        self.render_thread = threading.Thread(target=self.render_loop, daemon=True)
        self.render_thread.start()
    
    def stop(self):
        if not self.running:
            return
        self.stop_event.set()
        try:
            if self.cap_thread: self.cap_thread.join(timeout=1.0)
            if self.proc_thread: self.proc_thread.join(timeout=1.0)
        except:
            pass
        try:
            if self.ndi_sender:
                ndi.send_destroy(self.ndi_sender)
                ndi.destroy()
        except:
            pass
        try:
            if self.virtual_cam:
                self.virtual_cam.close()
        except:
            pass
        if self.pose_tracker:
            self.pose_tracker.release()
        self.running = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_var.set("Стриминг остановлен")
        gc.collect()
    
    def quit(self):
        if self.running:
            if not messagebox.askyesno("Quit", "Остановить стриминг и выйти?"):
                return
            self.stop()
        self.root.quit()
    
    def update_preview(self, frame):
        try:
            preview_frame = cv2.resize(frame, (640, 360))
            preview_frame = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
            img = tk.PhotoImage(data=cv2.imencode('.png', preview_frame)[1].tobytes())
            self.preview_label.configure(image=img)
            self.preview_label.image = img
        except Exception as e:
            print(f"Preview update error: {e}")
    
    def render_loop(self):
        last_frame = np.zeros((self.WINDOW_H, self.WINDOW_W, 3), dtype=np.uint8)
        last_pose_data = None
        last_barbell_pos = None
        last_send = 0.0
        frame_counter = 0
        LEFT_WRIST, RIGHT_WRIST = 15, 16
        
        while not self.stop_event.is_set():
            frame_counter += 1
            try:
                frame, pose_data, barbell_pos, timestamp = self.render_q.get(timeout=0.05)
                last_frame = frame.copy()
                last_pose_data = pose_data
                last_barbell_pos = barbell_pos
            except queue.Empty:
                frame, pose_data, barbell_pos, timestamp = last_frame, last_pose_data, last_barbell_pos, time.time()
            
            disp = frame.copy()
            angles = {}
            left_knee_coords = None
            right_knee_coords = None
            left_knee_angle = None
            right_knee_angle = None
            joints_data = {}
            
            # Обработка позы
            if self.enable_pose.get() and pose_data and pose_data.get('all_landmarks'):
                lm = pose_data['all_landmarks']
                h, w = frame.shape[:2]
                
                # Вычисление углов
                try:
                    angles["left_arm"] = calculate_angle((lm[11].x,lm[11].y),(lm[13].x,lm[13].y),(lm[15].x,lm[15].y))
                    angles["right_arm"] = calculate_angle((lm[12].x,lm[12].y),(lm[14].x,lm[14].y),(lm[16].x,lm[16].y))
                    angles["left_leg"] = calculate_angle((lm[23].x,lm[23].y),(lm[25].x,lm[25].y),(lm[27].x,lm[27].y))
                    angles["right_leg"] = calculate_angle((lm[24].x,lm[24].y),(lm[26].x,lm[26].y),(lm[28].x,lm[28].y))
                except:
                    angles = {}
                
                # Сглаживание углов
                for k,v in angles.items():
                    buf = self.angle_buffer.get(k, [])
                    buf.append(v)
                    if len(buf) > 5:
                        buf.pop(0)
                    self.angle_buffer[k] = buf
                    angles[k] = sum(buf)/len(buf) if buf else v
                
                # Отрисовка скелета
                if self.show_joints.get():
                    disp = draw_overlay(disp, lm, angles, self.bone_color.get(), self.joint_color.get(),
                                      self.bone_width.get(), self.joint_radius.get())
                
                # Данные для UDP
                if pose_data.get('all_landmarks'):
                    joints_data = {str(i): [float(l.x), float(l.y), float(getattr(l, 'z', 0.0))]
                                 for i, l in enumerate(lm)}
                    joints = self.pose_tracker.get_leg_joints(pose_data) if self.pose_tracker else {}
                    left_knee_coords = joints.get('left_knee')
                    right_knee_coords = joints.get('right_knee')
                    left_knee_angle = joints.get('left_knee_angle')
                    right_knee_angle = joints.get('right_knee_angle')
            
            # Визуализация пути штанги (через Visualizer)
            if self.enable_barbell.get() and self.barbell_tracker and self.visualizer:
                disp = self.visualizer.draw_frame(disp, pose_data if self.enable_pose.get() else None,
                                                 barbell_pos, self.barbell_tracker.get_path())
            elif self.enable_barbell.get() and self.barbell_tracker:
                # Если visualizer не создан, рисуем путь вручную
                path = self.barbell_tracker.get_path()
                if len(path) >= 2:
                    for i in range(1, len(path)):
                        pt1 = (int(path[i-1][0]), int(path[i-1][1]))
                        pt2 = (int(path[i][0]), int(path[i][1]))
                        cv2.line(disp, pt1, pt2, config.COLOR_BARBELL_PATH, config.LINE_THICKNESS)
            
            # Отправка UDP данных
            udp_data = {
                "timestamp": timestamp,
                "barbell": {
                    "position": [int(barbell_pos[0]), int(barbell_pos[1])] if barbell_pos else None,
                    "confidence": float(self.barbell_tracker.last_confidence) if (self.barbell_tracker and self.barbell_tracker.last_confidence) else None,
                    "source": self.barbell_tracker.last_detection_source if self.barbell_tracker else None
                },
                "knee_positions": {
                    "left_knee": list(left_knee_coords) if left_knee_coords else None,
                    "right_knee": list(right_knee_coords) if right_knee_coords else None,
                    "left_knee_angle": float(left_knee_angle) if left_knee_angle else None,
                    "right_knee_angle": float(right_knee_angle) if right_knee_angle else None
                },
                "joints": joints_data,
                "barbell_path": [
                    {"x": float(x), "y": float(y), "timestamp": float(ts)}
                    for x, y, ts in (self.barbell_tracker.get_path() if self.barbell_tracker else [])
                ],
                "angles": {k: round(v,2) for k,v in angles.items()} if angles else {}
            }
            try:
                sock.sendto(json.dumps(udp_data, ensure_ascii=False).encode('utf-8'), (UE_IP, UE_PORT))
            except:
                pass
            
            # Изменение размера
            if disp.shape[1] != self.WINDOW_W or disp.shape[0] != self.WINDOW_H:
                disp = resize_with_aspect(disp, self.WINDOW_W, self.WINDOW_H)
            
            # Обновление предпросмотра
            self.root.after(0, self.update_preview, disp.copy())
            
            # Отображение
            cv2.imshow("Unified Tracking (ESC to stop)", disp)
            
            # Виртуальная камера
            if self.virtual_cam:
                try:
                    self.virtual_cam.send(disp)
                    self.virtual_cam.sleep_until_next_frame()
                except:
                    pass
            
            # NDI
            if self.ndi_sender:
                now = time.time()
                if now - last_send >= 1.0 / self.target_fps:
                    try:
                        bgrx = np.zeros((self.WINDOW_H, self.WINDOW_W, 4), dtype=np.uint8)
                        bgrx[:, :, :3] = disp
                        vf = ndi.VideoFrameV2()
                        vf.data = bgrx
                        vf.xres = self.WINDOW_W
                        vf.yres = self.WINDOW_H
                        vf.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRX
                        ndi.send_send_video_v2(self.ndi_sender, vf)
                    except Exception as e:
                        if frame_counter % 300 == 0:
                            print("NDI send error:", e)
                    last_send = now
            
            # Обработка горячих клавиш
            k = cv2.waitKey(1) & 0xFF
            if k == 27 or k == ord('q'):  # ESC или 'q' - остановка
                self.stop()
                break
            elif k == ord('c') or k == ord('C'):  # 'c' - очистить путь штанги
                if self.barbell_tracker:
                    self.barbell_tracker.clear_path()
                    print("Путь штанги очищен")
            elif k == ord('1'):  # '1' - переключить трекинг позы
                self.enable_pose.set(not self.enable_pose.get())
                print(f"Трекинг позы: {'ВКЛ' if self.enable_pose.get() else 'ВЫКЛ'}")
            elif k == ord('2'):  # '2' - переключить трекинг штанги
                self.enable_barbell.set(not self.enable_barbell.get())
                print(f"Трекинг штанги: {'ВКЛ' if self.enable_barbell.get() else 'ВЫКЛ'}")
        
        try:
            cv2.destroyAllWindows()
        except:
            pass

# -------------------- main --------------------
def main():
    root = tk.Tk()
    root.geometry("1200x700")
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.quit)
    root.mainloop()

if __name__ == "__main__":
    main()

