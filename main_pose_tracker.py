# pose_gui_ndistream.py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # тише логи TensorFlow/MediaPipe

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

# MediaPipe и параметры по умолчанию (pose создаётся асинхронно)
import mediapipe as mp
mp_pose = mp.solutions.pose
pose = None  # будет установлено асинхронно
pose_ready = threading.Event()

# UDP (Unreal) - используем config.py если доступен
try:
    import config
    UE_IP, UE_PORT = config.UDP_HOST, config.UDP_PORT
except ImportError:
    UE_IP, UE_PORT = "127.0.0.1", 5000
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# -------------------- утилиты --------------------
def list_cameras(max_test=6):
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
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return 0.0
    cosang = np.dot(ba, bc) / denom
    return float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))

def resize_with_aspect(frame, target_w, target_h):
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w*scale), int(h*scale)
    resized = cv2.resize(frame, (new_w, new_h))
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x = (target_w - new_w)//2
    y = (target_h - new_h)//2
    canvas[y:y+new_h, x:x+new_w] = resized
    return canvas

def default_packet():
    return {"pose_detected": False, "angles": {"left_arm":0,"right_arm":0,"left_leg":0,"right_leg":0}}

# -------------------- асинхронная инициализация Pose --------------------
def create_pose_model(model_complexity=0, smooth=True, min_det=0.5, min_track=0.5):
    return mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        smooth_landmarks=smooth,
        enable_segmentation=False,
        min_detection_confidence=min_det,
        min_tracking_confidence=min_track
    )

def init_pose_async(model_complexity, smooth, min_det, min_track):
    global pose
    pose = create_pose_model(model_complexity, smooth, min_det, min_track)
    pose_ready.set()
    print("✅ Pose model ready")

# -------------------- потоки: захват и обработка --------------------
class CaptureThread(threading.Thread):
    def __init__(self, source, out_q, stop_event):
        super().__init__(daemon=True)
        self.source = source
        self.out_q = out_q
        self.stop_event = stop_event
        self.cap = None
        self.open_source(source)

    def open_source(self, source):
        if isinstance(source, str) and source.lower().endswith(".mp4"):
            self.cap = cv2.VideoCapture(source)
        else:
            # если строка цифры — конвертим в int
            try:
                idx = int(source)
                self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW if os.name == "nt" else 0)
            except Exception:
                self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            print("❌ Cannot open source:", source)

    def switch_source(self, new_src):
        try:
            if self.cap: self.cap.release()
        except: pass
        self.open_source(new_src)

    def run(self):
        while not self.stop_event.is_set():
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.05)
                continue
            ret, frame = self.cap.read()
            if not ret:
                # если mp4 — прокручиваем в начало
                if isinstance(self.source, str) and self.source.lower().endswith(".mp4"):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.02)
                continue
            # flip to be natural
            frame = cv2.flip(frame, 1)
            # non-blocking put with drop oldest if full
            try:
                self.out_q.put(frame, block=False)
            except queue.Full:
                try:
                    _ = self.out_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.out_q.put(frame, block=False)
                except:
                    pass
        try:
            self.cap.release()
        except:
            pass

class ProcThread(threading.Thread):
    def __init__(self, in_q, out_q, stop_event, proc_w, proc_h, every_n):
        super().__init__(daemon=True)
        self.in_q = in_q
        self.out_q = out_q
        self.stop_event = stop_event
        self.proc_w = proc_w
        self.proc_h = proc_h
        self.every_n = every_n
        self.idx = 0

    def run(self):
        # ждем пока модель готова
        pose_ready.wait()
        while not self.stop_event.is_set():
            try:
                frame = self.in_q.get(timeout=0.05)
            except queue.Empty:
                time.sleep(0.01)
                continue
            self.idx += 1
            results = None
            if self.idx % self.every_n == 0:
                small = cv2.resize(frame, (self.proc_w, self.proc_h))
                small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                try:
                    results = pose.process(small_rgb)
                except Exception as e:
                    print("Pose process error:", e)
                    results = None
            # push tuple (frame (full), results)
            try:
                self.out_q.put((frame, results), block=False)
            except queue.Full:
                try:
                    _ = self.out_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.out_q.put((frame, results), block=False)
                except:
                    pass

# -------------------- отрисовка скелета --------------------
def draw_overlay(frame, landmarks, angles, bone_color, joint_color, bone_width, joint_radius):
    """Рисует линии и углы для рук и ног с настраиваемыми цветами и размерами."""
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

    # Конвертируем цвета из hex в BGR
    bone_bgr = tuple(int(bone_color[i:i+2], 16) for i in (5, 3, 1))
    joint_bgr = tuple(int(joint_color[i:i+2], 16) for i in (5, 3, 1))

    for limb, (a, b, c) in limbs.items():
        try:
            pa = (int(landmarks[a].x * w), int(landmarks[a].y * h))
            pb = (int(landmarks[b].x * w), int(landmarks[b].y * h))
            pc = (int(landmarks[c].x * w), int(landmarks[c].y * h))
        except:
            continue

        # линии костей с обводкой
        outline_width = bone_width + 2
        cv2.line(overlay, pa, pb, (0, 0, 0), outline_width, cv2.LINE_AA)
        cv2.line(overlay, pb, pc, (0, 0, 0), outline_width, cv2.LINE_AA)
        cv2.line(overlay, pa, pb, bone_bgr, bone_width, cv2.LINE_AA)
        cv2.line(overlay, pb, pc, bone_bgr, bone_width, cv2.LINE_AA)

        # углы - используем букву 'o' вместо символа градуса
        angle_val = angles.get(limb, 0.0)
        cv2.putText(
            overlay, f"{angle_val:.0f}o",
            (pb[0] + 10, pb[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            overlay, f"{angle_val:.0f}o",
            (pb[0] + 10, pb[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, bone_bgr, 1, cv2.LINE_AA
        )

        # суставы
        for idx in [a, b, c]:
            x, y = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
            cv2.circle(overlay, (x, y), joint_radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(overlay, (x, y), joint_radius, joint_bgr, -1, cv2.LINE_AA)

    return cv2.addWeighted(overlay, 0.9, frame, 0.1, 0)

# -------------------- GUI --------------------
class App:
    def __init__(self, root):
        self.root = root
        root.title("Pose Tracking System")
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

        # default params
        self.WINDOW_W = 1920
        self.WINDOW_H = 1080
        self.proc_w = 320
        self.proc_h = 180
        self.every_n = 2
        self.target_fps = 30
        self.ndi_name = tk.StringVar(value="PoseStream_NDI")
        self.use_ndi = tk.BooleanVar(value=NDI_AVAILABLE)
        self.use_virtual = tk.BooleanVar(value=False and VIRTUALCAM_AVAILABLE)
        self.show_joints = tk.BooleanVar(value=True)
        self.model_complexity = tk.IntVar(value=1)
        self.smooth_landmarks = tk.BooleanVar(value=True)
        self.min_det = tk.DoubleVar(value=0.4)
        self.min_track = tk.DoubleVar(value=0.4)
        
        # Новые параметры для цветов и размеров
        self.bone_color = tk.StringVar(value="#FF6B35")  # оранжевый по умолчанию
        self.joint_color = tk.StringVar(value="#4ECDC4")  # бирюзовый по умолчанию
        self.bone_width = tk.IntVar(value=6)
        self.joint_radius = tk.IntVar(value=6)

        # Основной layout с двумя колонками
        main_frame = ttk.Frame(root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Левая колонка - настройки
        left_frame = ttk.Frame(main_frame, width=400)
        left_frame.pack(side='left', fill='y', padx=(0, 10))
        left_frame.pack_propagate(False)

        # Правая колонка - предпросмотр
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side='right', fill='both', expand=True)

        # === ПРЕДПРОСМОТР ===
        preview_header = ttk.Label(right_frame, text="Предпросмотр", font=('Arial', 12, 'bold'))
        preview_header.pack(pady=(0, 5))
        
        preview_container = ttk.Frame(right_frame)
        preview_container.pack(fill='both', expand=True)
        
        self.preview_label = ttk.Label(preview_container, text="Запустите стрим для предпросмотра", 
                                      background='black', foreground='white', 
                                      font=('Arial', 10), anchor='center')
        self.preview_label.pack(fill='both', expand=True)

        # === НАСТРОЙКИ ИСТОЧНИКА ===
        source_frame = ttk.LabelFrame(left_frame, text="📷 Источник видео", padding=10)
        source_frame.pack(fill='x', pady=(0, 10))

        source_row1 = ttk.Frame(source_frame)
        source_row1.pack(fill='x', pady=2)
        ttk.Label(source_row1, text="Камера/Файл:").pack(side='left')
        self.cam_list = list_cameras(6)
        self.source_var = tk.StringVar(value=str(self.cam_list[0]) if self.cam_list else "0")
        self.source_combo = ttk.Combobox(source_row1, values=[str(x) for x in self.cam_list], 
                                       textvariable=self.source_var, width=12)
        self.source_combo.pack(side='left', padx=5)

        source_row2 = ttk.Frame(source_frame)
        source_row2.pack(fill='x', pady=2)
        ttk.Button(source_row2, text="Выбрать MP4", command=self.browse_file).pack(side='left', padx=2)
        ttk.Button(source_row2, text="Обновить камеры", command=self.refresh_cams).pack(side='left', padx=2)

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
        self.every_spin = ttk.Spinbox(proc_row1, from_=1, to=6, width=4, 
                                    textvariable=tk.IntVar(value=self.every_n))
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

        # === НАСТРОЙКИ ВНЕШНЕГО ВИДА ===
        appearance_frame = ttk.LabelFrame(left_frame, text="🎨 Внешний вид скелета", padding=10)
        appearance_frame.pack(fill='x', pady=(0, 10))

        # Цвета
        colors_frame = ttk.Frame(appearance_frame)
        colors_frame.pack(fill='x', pady=5)

        ttk.Label(colors_frame, text="Цвет костей:").grid(row=0, column=0, sticky="w", padx=5, pady=3)
        self.bone_color_btn = ttk.Button(colors_frame, text="Выбрать", command=self.choose_bone_color, width=8)
        self.bone_color_btn.grid(row=0, column=1, padx=5, pady=3)
        self.bone_color_preview = tk.Canvas(colors_frame, width=40, height=20, bg=self.bone_color.get(), 
                                          relief='solid', bd=1)
        self.bone_color_preview.grid(row=0, column=2, padx=5, pady=3)

        ttk.Label(colors_frame, text="Цвет суставов:").grid(row=1, column=0, sticky="w", padx=5, pady=3)
        self.joint_color_btn = ttk.Button(colors_frame, text="Выбрать", command=self.choose_joint_color, width=8)
        self.joint_color_btn.grid(row=1, column=1, padx=5, pady=3)
        self.joint_color_preview = tk.Canvas(colors_frame, width=40, height=20, bg=self.joint_color.get(),
                                           relief='solid', bd=1)
        self.joint_color_preview.grid(row=1, column=2, padx=5, pady=3)

        # Размеры
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

        # === НАСТРОЙКИ МОДЕЛИ ===
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

        # === ВЫХОДНЫЕ НАСТРОЙКИ ===
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

        # internal
        self.cap_thread = None
        self.proc_thread = None
        self.proc_q = None
        self.render_q = None
        self.ndi_sender = None
        self.virtual_cam = None
        self.last_frame = np.zeros((self.WINDOW_W, self.WINDOW_H, 3), dtype=np.uint8)
        self.last_results = None
        self.angle_buffer = {k: [] for k in ["left_arm","right_arm","left_leg","right_leg"]}
        self.frame_counter = 0

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
        path = filedialog.askopenfilename(filetypes=[("MP4 files","*.mp4"),("All files","*.*")])
        if path:
            self.source_var.set(path)

    def refresh_cams(self):
        cams = list_cameras(8)
        self.source_combo['values'] = [str(x) for x in cams]
        if cams:
            self.source_var.set(str(cams[0]))

    def start(self):
        if self.running:
            return
        # read params
        src = self.source_var.get()
        proc_res = self.proc_entry.get().strip()
        try:
            pw, ph = [int(x) for x in proc_res.split("x")]
        except Exception:
            messagebox.showerror("Error", "Proc resolution must be like 320x180")
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

        # model init async
        self.status_var.set("Инициализация модели...")
        pose_ready.clear()
        t = threading.Thread(target=init_pose_async, args=(self.model_complexity.get(), self.smooth_landmarks.get(), self.min_det.get(), self.min_track.get()), daemon=True)
        t.start()
        # prepare queues and threads
        self.proc_q = queue.Queue(maxsize=2)
        self.render_q = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()

        self.cap_thread = CaptureThread(src, self.proc_q, self.stop_event)
        self.proc_thread = ProcThread(self.proc_q, self.render_q, self.stop_event, self.proc_w, self.proc_h, self.every_n)
        self.cap_thread.start()
        self.proc_thread.start()

        # wait until pose ready (with timeout)
        waited = pose_ready.wait(timeout=10.0)
        if not waited:
            messagebox.showwarning("Warning", "Pose model is taking long to initialize. Continue anyway.")
        # init NDI if selected
        if self.use_ndi.get() and NDI_AVAILABLE:
            try:
                if not ndi.initialize():
                    messagebox.showwarning("NDI", "NDI.initialize() returned False")
                else:
                    sc = ndi.SendCreate()
                    sc.ndi_name = self.ndi_name.get()
                    self.ndi_sender = ndi.send_create(sc)
                    if self.ndi_sender:
                        print("NDI sender created:", sc.ndi_name)
            except Exception as e:
                messagebox.showwarning("NDI", f"NDI init error: {e}")
                self.ndi_sender = None
        # virtual cam
        if self.use_virtual.get() and VIRTUALCAM_AVAILABLE:
            try:
                self.virtual_cam = pyvirtualcam.Camera(width=self.WINDOW_W, height=self.WINDOW_H, fps=self.target_fps, fmt=PixelFormat.BGR)
            except Exception as e:
                messagebox.showwarning("VirtualCam", f"Error creating virtual cam: {e}")
                self.virtual_cam = None

        self.running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_var.set("Стриминг активен")
        # start render loop in background
        self.render_thread = threading.Thread(target=self.render_loop, daemon=True)
        self.render_thread.start()

    def stop(self):
        if not self.running:
            return
        self.stop_event.set()
        # join threads gracefully
        try:
            if self.cap_thread: self.cap_thread.join(timeout=1.0)
            if self.proc_thread: self.proc_thread.join(timeout=1.0)
        except:
            pass
        # destroy NDI and virtual cam
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
        self.running = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_var.set("Стриминг остановлен")
        gc.collect()

    def quit(self):
        if self.running:
            if not messagebox.askyesno("Quit", "Stop streaming and quit?"):
                return
            self.stop()
        self.root.quit()

    def update_preview(self, frame):
        """Обновляет предпросмотр в GUI"""
        try:
            # Масштабируем кадр для предпросмотра
            preview_frame = cv2.resize(frame, (640, 360))
            preview_frame = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
            
            # Конвертируем в PhotoImage
            img = tk.PhotoImage(data=cv2.imencode('.png', preview_frame)[1].tobytes())
            self.preview_label.configure(image=img)
            self.preview_label.image = img  # сохраняем ссылку
        except Exception as e:
            print(f"Preview update error: {e}")

    def render_loop(self):
        # display loop: consume from render_q, show last frame if none
        last_frame = np.zeros((self.WINDOW_H, self.WINDOW_W, 3), dtype=np.uint8)
        last_results = None
        last_send = 0.0
        frame_counter = 0
        while not self.stop_event.is_set():
            frame_counter += 1
            try:
                frame, results = self.render_q.get(timeout=0.05)
                last_frame = frame.copy()
                last_results = results
            except queue.Empty:
                frame, results = last_frame, last_results

            # compute (use last_results if results is None)
            if results and getattr(results, "pose_landmarks", None):
                lm = results.pose_landmarks.landmark
            elif last_results and getattr(last_results, "pose_landmarks", None):
                lm = last_results.pose_landmarks.landmark
            else:
                lm = None

            disp = frame.copy()
            # angles smoothing buffer
            angles = {}
            if lm:
                try:
                    angles["left_arm"] = calculate_angle((lm[11].x,lm[11].y),(lm[13].x,lm[13].y),(lm[15].x,lm[15].y))
                    angles["right_arm"] = calculate_angle((lm[12].x,lm[12].y),(lm[14].x,lm[14].y),(lm[16].x,lm[16].y))
                    angles["left_leg"] = calculate_angle((lm[23].x,lm[23].y),(lm[25].x,lm[25].y),(lm[27].x,lm[27].y))
                    angles["right_leg"] = calculate_angle((lm[24].x,lm[24].y),(lm[26].x,lm[26].y),(lm[28].x,lm[28].y))
                except Exception:
                    angles = {}
                # simple moving average (buffer length 5)
                for k,v in angles.items():
                    buf = self.angle_buffer.get(k, [])
                    buf.append(v)
                    if len(buf) > 5:
                        buf.pop(0)
                    self.angle_buffer[k] = buf
                    angles[k] = sum(buf)/len(buf) if buf else v
                # draw overlay (only if show_joints enabled)
                if self.show_joints.get():
                    # используем новые параметры цветов и размеров
                    disp = draw_overlay(disp, lm, angles, 
                                      self.bone_color.get(), 
                                      self.joint_color.get(),
                                      self.bone_width.get(),
                                      self.joint_radius.get())
                # send JSON
                packet = default_packet()
                packet["pose_detected"] = True
                packet["angles"] = {k: round(v,2) for k,v in angles.items()}
                try:
                    sock.sendto(json.dumps(packet).encode("utf-8"), (UE_IP, UE_PORT))
                except:
                    pass

            # resize to window
            if disp.shape[1] != self.WINDOW_W or disp.shape[0] != self.WINDOW_H:
                disp = resize_with_aspect(disp, self.WINDOW_W, self.WINDOW_H)

            # optional overlay FPS small
            #cv2.putText(disp, f"FPS target:{self.target_fps}", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)

            # Обновляем предпросмотр в GUI
            self.root.after(0, self.update_preview, disp.copy())

            # show
            cv2.imshow("Pose Stream (press ESC to stop)", disp)
            # virtual cam
            if self.virtual_cam:
                try:
                    self.virtual_cam.send(disp)
                    self.virtual_cam.sleep_until_next_frame()
                except Exception:
                    pass

            # NDI send throttled to target_fps
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
                        # log rarely
                        if frame_counter % 300 == 0:
                            print("NDI send error:", e)
                    last_send = now

            # handle local ESC to stop
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                self.stop()
                break
        # end loop
        try:
            cv2.destroyAllWindows()
        except:
            pass

# -------------------- main --------------------
def main():
    root = tk.Tk()
    root.geometry("1200x700")  # Устанавливаем начальный размер окна
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.quit)
    root.mainloop()

if __name__ == "__main__":
    main()