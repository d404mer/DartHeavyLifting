"""
Объединенная версия: GUI + трекинг позы + трекинг штанги
С GUI вынесенным в отдельный файл
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
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional, Tuple, List
from collections import deque
from urllib.parse import parse_qs, unquote
from PIL import ImageFont, ImageDraw, Image

# Импорты из проекта
import config
from pose_tracker import PoseTracker
from visualizer import Visualizer

# Импортируем только GUI из отдельного файла
from gui import AppGUI

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

# -------------------- Утилиты --------------------
def list_cameras(max_test=6):
    """Список доступных камер через OpenCV"""
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
    def __init__(self, source, out_q, stop_event, target_fps=50):
        super().__init__(daemon=True)
        self.source = source
        self.out_q = out_q
        self.stop_event = stop_event
        self.target_fps = target_fps
        self.cap = None
        self.is_video_file = False
        self.video_fps = 50.0
        self.use_ffmpeg = False
        self.ffmpeg_process = None
        self.ffmpeg_width = getattr(config, "VIDEO_WIDTH", 1920)
        self.ffmpeg_height = getattr(config, "VIDEO_HEIGHT", 1080)
        self.ffmpeg_pixel_format = getattr(config, "DECKLINK_DEFAULT_PIXEL_FORMAT", "bgr24")
        self.ffmpeg_frame_size = self.ffmpeg_width * self.ffmpeg_height * 3
        self.ffmpeg_stderr_thread = None
        self.open_source(source)
    
    def open_source(self, source):
        # Захват через ffmpeg (DeckLink)
        if isinstance(source, str) and source.lower().startswith("decklink:"):
            try:
                self._start_decklink_capture(source)
            except Exception as e:
                self.use_ffmpeg = False
                print(f"❌ Не удалось запустить ffmpeg для источника '{source}': {e}")
            return
        
        # Обычный видеофайл
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
            return
        
        # Попытка открыть числовой индекс (DirectShow/Media Foundation)
        try:
            idx = int(source)
            self.is_video_file = False
            
            # Получаем параметры из config
            target_width = getattr(config, "VIDEO_WIDTH", 1920)
            target_height = getattr(config, "VIDEO_HEIGHT", 1080)
            target_fps = getattr(config, "TARGET_FPS", 50)
            
            # Пробуем DirectShow сначала
            self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW if os.name == "nt" else 0)
            
            if self.cap.isOpened():
                # Пробуем установить параметры
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
                self.cap.set(cv2.CAP_PROP_FPS, target_fps)
                
                # Пропускаем несколько кадров для стабилизации
                for _ in range(5):
                    ret, _ = self.cap.read()
                    if not ret:
                        break
                
                # Проверяем реальные параметры
                actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
                
                print(f"✅ Камера {idx} открыта через DirectShow")
                print(f"   Разрешение: {actual_width}x{actual_height}, FPS: {actual_fps:.2f}")
            else:
                # Если DirectShow не сработал, пробуем Media Foundation
                try:
                    self.cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
                    if self.cap.isOpened():
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
                        self.cap.set(cv2.CAP_PROP_FPS, target_fps)
                        
                        for _ in range(5):
                            ret, _ = self.cap.read()
                            if not ret:
                                break
                        
                        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
                        
                        print(f"✅ Камера {idx} открыта через Media Foundation")
                        print(f"   Разрешение: {actual_width}x{actual_height}, FPS: {actual_fps:.2f}")
                except Exception as e:
                    print(f"⚠️ Media Foundation не сработал: {e}")
        except Exception:
            self.cap = cv2.VideoCapture(source)
            self.is_video_file = False
        
        if not self.use_ffmpeg and (self.cap is None or not self.cap.isOpened()):
            print("❌ Cannot open source:", source)
    
    def _start_decklink_capture(self, source: str):
        """Инициализация захвата через ffmpeg с backend DeckLink"""
        self.use_ffmpeg = True
        self.is_video_file = False
        
        spec = source[len("decklink:"):]
        if "?" in spec:
            device_part, query_part = spec.split("?", 1)
            params = parse_qs(query_part, keep_blank_values=True)
        else:
            device_part = spec
            params = {}
        
        device_name = unquote(device_part).strip()
        if not device_name:
            device_name = getattr(config, "DECKLINK_DEFAULT_DEVICE", None)
        if not device_name:
            device_name = "0"
        
        # Параметры потока
        self.ffmpeg_width = int(params.get("width", [getattr(config, "VIDEO_WIDTH", 1920)])[0])
        self.ffmpeg_height = int(params.get("height", [getattr(config, "VIDEO_HEIGHT", 1080)])[0])
        fps_param = params.get("fps") or params.get("framerate")
        ffmpeg_fps = None
        if fps_param:
            try:
                ffmpeg_fps = float(fps_param[0])
            except (ValueError, TypeError):
                ffmpeg_fps = None
        format_code = params.get("format_code", [getattr(config, "DECKLINK_DEFAULT_FORMAT_CODE", None)])[0]
        
        pixel_format = params.get("pix_fmt", [getattr(config, "DECKLINK_DEFAULT_PIXEL_FORMAT", "bgr24")])[0]
        pixel_format = (pixel_format or "bgr24").lower()
        if pixel_format != "bgr24":
            print(f"⚠️ Поддерживается только вывод bgr24. Запрошен '{pixel_format}', использую 'bgr24'.")
            pixel_format = "bgr24"
        self.ffmpeg_pixel_format = pixel_format
        self.ffmpeg_frame_size = self.ffmpeg_width * self.ffmpeg_height * 3
        
        ffmpeg_path = getattr(config, "FFMPEG_PATH", "ffmpeg")
        cmd = [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel", "error",
            "-nostdin",
            "-thread_queue_size", "2048",
            "-f", "decklink",
        ]
        if format_code:
            cmd.extend(["-format_code", format_code])
        if ffmpeg_fps:
            cmd.extend(["-framerate", str(ffmpeg_fps)])
        cmd.extend(["-i", device_name])
        cmd.extend([
            "-pix_fmt", pixel_format,
            "-vsync", "0",
            "-f", "rawvideo",
            "-"
        ])
        
        try:
            self.ffmpeg_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )
        except FileNotFoundError:
            raise RuntimeError(f"FFmpeg не найден по пути '{ffmpeg_path}'. Установите сборку с поддержкой DeckLink.")
        except Exception as exc:
            raise RuntimeError(f"Ошибка запуска FFmpeg: {exc}")
        
        self.ffmpeg_stderr_thread = threading.Thread(target=self._consume_ffmpeg_stderr, daemon=True)
        self.ffmpeg_stderr_thread.start()
        fps_info = ffmpeg_fps if ffmpeg_fps else getattr(config, "TARGET_FPS", 30)
        print(f"🎥 FFmpeg DeckLink: '{device_name}' -> {self.ffmpeg_width}x{self.ffmpeg_height}@{fps_info}fps")
    
    def _consume_ffmpeg_stderr(self):
        """Вывод предупреждений ffmpeg, чтобы не переполнялся буфер stderr"""
        if not self.ffmpeg_process or self.ffmpeg_process.stderr is None:
            return
        try:
            for raw_line in self.ffmpeg_process.stderr:
                if not raw_line:
                    break
                try:
                    line = raw_line.decode("utf-8", "ignore").strip()
                except Exception:
                    line = str(raw_line).strip()
                if line:
                    print(f"[ffmpeg] {line}")
        except Exception:
            pass
    
    def _cleanup_capture(self):
        """Освобождение ресурсов захвата"""
        if self.use_ffmpeg:
            if self.ffmpeg_process:
                try:
                    if self.ffmpeg_process.stdout:
                        self.ffmpeg_process.stdout.close()
                    if self.ffmpeg_process.stderr:
                        self.ffmpeg_process.stderr.close()
                except Exception:
                    pass
                try:
                    self.ffmpeg_process.terminate()
                    self.ffmpeg_process.wait(timeout=2.0)
                except Exception:
                    try:
                        self.ffmpeg_process.kill()
                    except Exception:
                        pass
            self.ffmpeg_process = None
        else:
            try:
                if self.cap:
                    self.cap.release()
            except Exception:
                pass
        self.cap = None
    def run(self):
        frame_time = 1.0 / self.video_fps if self.is_video_file else 1.0 / max(self.target_fps, 1)
        last_frame_time = time.time()
        
        try:
            while not self.stop_event.is_set():
                if self.use_ffmpeg:
                    if not self.ffmpeg_process or self.ffmpeg_process.stdout is None:
                        if self.ffmpeg_process and self.ffmpeg_process.poll() is not None:
                            print("❌ FFmpeg DeckLink: процесс завершился")
                            break
                        time.sleep(0.05)
                        continue
                    
                    data = self.ffmpeg_process.stdout.read(self.ffmpeg_frame_size)
                    if not data or len(data) < self.ffmpeg_frame_size:
                        if self.stop_event.is_set():
                            break
                        if self.ffmpeg_process and self.ffmpeg_process.poll() is not None:
                            print("❌ FFmpeg DeckLink: поток остановлен")
                            break
                        time.sleep(0.01)
                        continue
                    
                    frame = np.frombuffer(data, dtype=np.uint8)
                    try:
                        frame = frame.reshape((self.ffmpeg_height, self.ffmpeg_width, 3))
                    except ValueError:
                        # Непредвиденный размер кадра
                        print("⚠️ FFmpeg DeckLink: Размер кадра не совпадает с ожидаемым")
                        time.sleep(0.01)
                        continue
                else:
                    if self.cap is None or not self.cap.isOpened():
                        time.sleep(0.05)
                        continue
                    
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
                    except Exception:
                        pass
        finally:
            self._cleanup_capture()

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
def draw_overlay(frame, landmarks, angles, bone_color, joint_color, bone_width, joint_radius, font_size=0.7, font_thickness=1):
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
            pa = (int(landmarks[a].x * w+480), int(landmarks[a].y * h))
            pb = (int(landmarks[b].x * w+480), int(landmarks[b].y * h))
            pc = (int(landmarks[c].x * w+480), int(landmarks[c].y * h))
        except:
            continue
        
        outline_width = bone_width + 2
        cv2.line(overlay, pa, pb, (0, 0, 0), outline_width, cv2.LINE_AA)
        cv2.line(overlay, pb, pc, (0, 0, 0), outline_width, cv2.LINE_AA)
        cv2.line(overlay, pa, pb, bone_bgr, bone_width, cv2.LINE_AA)
        cv2.line(overlay, pb, pc, bone_bgr, bone_width, cv2.LINE_AA)
        
        angle_val = angles.get(limb, 0.0)
        
        # Используем настройки шрифта из GUI
        outline_thickness = max(2, font_thickness)  # Обводка толще основного текста
        
        # Рисуем обводку текста
        # cv2.putText(overlay, f"{angle_val:.0f}", (pb[0] + 10, pb[1] - 10),
                   # cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), outline_thickness, cv2.LINE_AA)



        image_pil = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGRA2RGB))
        draw = ImageDraw.Draw(image_pil)
        font = ImageFont.truetype("arial.ttf",font_size*50) if hasattr(ImageFont, 'truetype') else ImageFont.load_default()
        if outline_thickness > 1:
            for dx in [-outline_thickness, 0, outline_thickness]:
                for dy in [-outline_thickness, 0, outline_thickness]:
                    if dx != 0 or dy != 0:
                        draw.text((pb[0] + 10 + 100, pb[1] - 10), f"{angle_val:.0f}°", font=font, fill=(255,255,255))
        draw.text((pb[0] +10 + 100, pb[1] - 10), f"{angle_val:.0f}°", font=font, fill=(255,255,255))   
        overlay  = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        # °
        # Рисуем основной текст
        # cv2.putText(overlay, f"{angle_val:.0f}", (pb[0] + 10, pb[1] - 10),
          #          cv2.FONT_HERSHEY_SIMPLEX, font_size, bone_bgr, font_thickness, cv2.LINE_AA)
        
        for idx in [a, b, c]:
            x, y = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
            cv2.circle(overlay, (x+480, y), joint_radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(overlay, (x+480, y), joint_radius, joint_bgr, -1, cv2.LINE_AA)
    
    return cv2.addWeighted(overlay, 0.9, frame, 0.1, 0)
# -------------------- Основной класс приложения --------------------
class UnifiedTrackingApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Unified Pose & Barbell Tracking")
        # Размеры окна
        window_width = 1200
        window_height = 1100
        
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        # Устанавливаем геометрию с позиционированием
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # Получаем список камер
        self.camera_list = list_cameras()
    
        
        # Создаем GUI
        self.gui = AppGUI(self.root, self.camera_list)
        
        # Устанавливаем callbacks
        self.gui.set_start_callback(self.start_processing)
        self.gui.set_stop_callback(self.stop_processing)
        self.gui.set_quit_callback(self.quit_app)
        self.gui.set_refresh_cameras_callback(self.refresh_cameras)
        
        # Состояние приложения
        self.running = False
        self.stop_event = threading.Event()
        
        # Потоки и очереди
        self.cap_thread = None
        self.proc_thread = None
        self.render_thread = None
        self.proc_q = None
        self.render_q = None
        
        # Трекеры
        self.pose_tracker = None
        self.barbell_tracker = None
        self.visualizer = None
        
        # Дополнительные выходы
        self.ndi_sender = None
        self.virtual_cam = None
        
        # Параметры окна
        self.WINDOW_W = 1920
        self.WINDOW_H = 1080
        
        # Угловой буфер для сглаживания
        self.angle_buffer = {k: [] for k in ["left_arm","right_arm","left_leg","right_leg"]}
        
        # UDP
        try:
            self.ue_ip, self.ue_port = config.UDP_HOST, config.UDP_PORT
        except:
            self.ue_ip, self.ue_port = "127.0.0.1", 5005
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Обработка закрытия окна
        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)
        
    def refresh_cameras(self):
        """Обновление списка камер"""
        self.camera_list = list_cameras()
        self.gui.update_camera_list(self.camera_list)
        
    def start_processing(self):
        """Запуск обработки видео"""
        if self.running:
            return
            
        try:
            # Получаем параметры из GUI
            source = self.gui.get_source()
            proc_w, proc_h, every_n, target_fps = self.gui.get_processing_params()
            
        except ValueError as e:
            messagebox.showerror("Ошибка", str(e))
            return
            
        # Инициализация трекеров
        self.gui.update_status("Инициализация...")
        
        # Трекер позы
        if self.gui.enable_pose.get():
            self.pose_tracker = PoseTracker(
                min_detection_confidence=self.gui.min_det.get(),
                min_tracking_confidence=self.gui.min_track.get()
            )
            
        # Трекер штанги
        if self.gui.enable_barbell.get():
            self.barbell_tracker = OptimizedBarbellTracker(smoothing_factor=config.BARBELL_SMOOTHING_FACTOR)
            
        # Визуализатор
        if self.gui.enable_barbell.get() or self.gui.enable_pose.get():
            self.visualizer = Visualizer(pose_tracker=self.pose_tracker if self.gui.enable_pose.get() else None)
            
        # Инициализация очередей и потоков
        self.proc_q = queue.Queue(maxsize=2)
        self.render_q = queue.Queue(maxsize=2)
        self.stop_event.clear()
        
        # Поток захвата
        self.cap_thread = CaptureThread(source, self.proc_q, self.stop_event, target_fps)
        
        # Поток обработки
        self.proc_thread = ProcThread(
            self.proc_q, self.render_q, self.stop_event, proc_w, proc_h, every_n,
            self.pose_tracker, self.barbell_tracker, self.gui.enable_barbell.get()
        )
        
        # Запускаем потоки
        self.cap_thread.start()
        self.proc_thread.start()
        
        # Инициализация дополнительных выходов
        self._initialize_outputs()
        
        # Обновляем состояние
        self.running = True
        self.gui.set_running_state(True)
        self.gui.update_status("Стриминг активен")
        
        # Запускаем поток рендеринга
        self.render_thread = threading.Thread(target=self._render_loop, daemon=True)
        self.render_thread.start()
        
        # Показываем горячие клавиши
        self._show_hotkeys()
        
    def stop_processing(self):
        """Остановка обработки видео"""
        if not self.running:
            return
            
        self.stop_event.set()
        
        # Останавливаем потоки
        if self.cap_thread:
            self.cap_thread.join(timeout=1.0)
        if self.proc_thread:
            self.proc_thread.join(timeout=1.0)
            
        # Закрываем дополнительные выходы
        self._cleanup_outputs()
        
        # Освобождаем ресурсы трекеров
        if self.pose_tracker:
            self.pose_tracker.release()
            
        # Обновляем состояние
        self.running = False
        self.gui.set_running_state(False)
        self.gui.update_status("Стриминг остановлен")
        
        # Очищаем предпросмотр
        self.gui.preview_label.configure(
            text="Запустите стрим для предпросмотра",
            image=''
        )
        
        gc.collect()
        
    def _initialize_outputs(self):
        """Инициализация дополнительных выходов (NDI, VirtualCam)"""
        # NDI
        if self.gui.use_ndi.get() and NDI_AVAILABLE:
            try:
                if ndi.initialize():
                    sc = ndi.SendCreate()
                    sc.ndi_name = self.gui.ndi_name.get()
                    self.ndi_sender = ndi.send_create(sc)
            except Exception as e:
                messagebox.showwarning("NDI", f"NDI init error: {e}")
                
        # Virtual Camera
        if self.gui.use_virtual.get() and VIRTUALCAM_AVAILABLE:
            try:
                self.virtual_cam = pyvirtualcam.Camera(
                    width=self.WINDOW_W, 
                    height=self.WINDOW_H,
                    fps=50,
                    fmt=PixelFormat.BGR
                )
            except Exception as e:
                messagebox.showwarning("VirtualCam", f"Error: {e}")
                
    def _cleanup_outputs(self):
        """Очистка дополнительных выходов"""
        # NDI
        try:
            if self.ndi_sender:
                ndi.send_destroy(self.ndi_sender)
                ndi.destroy()
                self.ndi_sender = None
        except:
            pass
            
        # Virtual Camera
        try:
            if self.virtual_cam:
                self.virtual_cam.close()
                self.virtual_cam = None
        except:
            pass
            
    def _show_hotkeys(self):
        """Показ информации о горячих клавишах"""
        print("\n" + "="*50)
        print("🎮 ГОРЯЧИЕ КЛАВИШИ (в окне OpenCV):")
        print("  'q' или ESC - остановка")
        print("  'c' - очистить путь штанги")
        print("  '1' - переключить трекинг позы")
        print("  '2' - переключить трекинг штанги")
        print("="*50 + "\n")
        
    def _render_loop(self):
        """Основной цикл рендеринга"""
        last_frame = np.zeros((self.WINDOW_H, self.WINDOW_W, 3), dtype=np.uint8)
        last_pose_data = None
        last_barbell_pos = None
        last_send = 0.0
        frame_counter = 0
        LEFT_WRIST, RIGHT_WRIST = 15, 16
        
        while not self.stop_event.is_set():
            frame_counter += 1
            
            try:
                # Получаем данные из очереди
                frame, pose_data, barbell_pos, timestamp = self.render_q.get(timeout=0.05)
                last_frame = frame.copy()
                last_pose_data = pose_data
                last_barbell_pos = barbell_pos
            except queue.Empty:
                # Используем последние данные если очередь пуста
                frame, pose_data, barbell_pos, timestamp = last_frame, last_pose_data, last_barbell_pos, time.time()
                
            display_frame = frame.copy()
            angles = {}
            left_knee_coords = None
            right_knee_coords = None
            left_knee_angle = None
            right_knee_angle = None
            joints_data = {}
            
            # Обработка позы
            if self.gui.enable_pose.get() and pose_data and pose_data.get('all_landmarks'):
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
                if self.gui.show_joints.get():
                    font_settings = self.gui.get_font_settings()
                    display_frame = draw_overlay(
                        display_frame, lm, angles, 
                        self.gui.bone_color.get(), 
                        self.gui.joint_color.get(),
                        self.gui.bone_width.get(), 
                        self.gui.joint_radius.get(),
                        font_size=font_settings['font_size'],
                        font_thickness=font_settings['font_thickness']
                    )
                
                # Данные для UDP
                if pose_data.get('all_landmarks'):
                    joints_data = {str(i): [float(l.x), float(l.y), float(getattr(l, 'z', 0.0))]
                                 for i, l in enumerate(lm)}
                    joints = self.pose_tracker.get_leg_joints(pose_data) if self.pose_tracker else {}
                    left_knee_coords = joints.get('left_knee')
                    right_knee_coords = joints.get('right_knee')
                    left_knee_angle = joints.get('left_knee_angle')
                    right_knee_angle = joints.get('right_knee_angle')
            
            # Визуализация пути штанги
            if self.gui.enable_barbell.get() and self.barbell_tracker and self.visualizer:
                display_frame = self.visualizer.draw_frame(
                    display_frame, 
                    pose_data if self.gui.enable_pose.get() else None,
                    barbell_pos, 
                    self.barbell_tracker.get_path()
                )
            elif self.gui.enable_barbell.get() and self.barbell_tracker:
                # Если visualizer не создан, рисуем путь вручную
                path = self.barbell_tracker.get_path()
                if len(path) >= 2:
                    for i in range(1, len(path)):
                        pt1 = (int(path[i-1][0]), int(path[i-1][1]))
                        pt2 = (int(path[i][0]), int(path[i][1]))
                        cv2.line(display_frame, pt1, pt2, config.COLOR_BARBELL_PATH, config.LINE_THICKNESS)
            
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
                self.sock.sendto(json.dumps(udp_data, ensure_ascii=False).encode('utf-8'), (self.ue_ip, self.ue_port))
            except:
                pass
            
            # Изменение размера
            if display_frame.shape[1] != self.WINDOW_W or display_frame.shape[0] != self.WINDOW_H:
                display_frame = resize_with_aspect(display_frame, self.WINDOW_W, self.WINDOW_H)
            
            # Обновление предпросмотра
            self.root.after(0, self.gui.update_preview, display_frame.copy())
            
            # Отображение
            cv2.imshow("Unified Tracking (ESC to stop)", display_frame)
            
            # Виртуальная камера
            if self.virtual_cam:
                try:
                    self.virtual_cam.send(display_frame)
                    self.virtual_cam.sleep_until_next_frame()
                except:
                    pass
            
            # NDI
            if self.ndi_sender:
                now = time.time()
                if now - last_send >= 1.0 / 120:  # 30 FPS для NDI
                    try:
                        bgrx = np.zeros((self.WINDOW_H, self.WINDOW_W, 4), dtype=np.uint8)
                        bgrx[:, :, :3] = display_frame
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
                self.stop_processing()
                break
            elif k == ord('c') or k == ord('C'):  # 'c' - очистить путь штанги
                if self.barbell_tracker:
                    self.barbell_tracker.clear_path()
                    print("Путь штанги очищен")
            elif k == ord('1'):  # '1' - переключить трекинг позы
                self.gui.enable_pose.set(not self.gui.enable_pose.get())
                print(f"Трекинг позы: {'ВКЛ' if self.gui.enable_pose.get() else 'ВЫКЛ'}")
            elif k == ord('2'):  # '2' - переключить трекинг штанги
                self.gui.enable_barbell.set(not self.gui.enable_barbell.get())
                print(f"Трекинг штанги: {'ВКЛ' if self.gui.enable_barbell.get() else 'ВЫКЛ'}")
        
        try:
            cv2.destroyAllWindows()
        except:
            pass
        
    def quit_app(self):
        """Выход из приложения"""
        if self.running:
            if not messagebox.askyesno("Выход", "Остановить стриминг и выйти?"):
                return
            self.stop_processing()
            
        self.root.quit()
        self.root.destroy()
        
    def run(self):
        """Запуск приложения"""
        self.root.mainloop()

def main():
    """Точка входа в приложение"""
    app = UnifiedTrackingApp()
    app.run()

if __name__ == "__main__":
    main()