import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.colorchooser import askcolor
from typing import Callable, Optional
import os
import json

class AppGUI:
    """Класс графического интерфейса приложения"""
    
    def __init__(self, root, camera_list: list):
        self.root = root
        self.camera_list = camera_list
        
        # Callbacks для взаимодействия с логикой приложения
        self.start_callback: Optional[Callable] = None
        self.stop_callback: Optional[Callable] = None
        self.quit_callback: Optional[Callable] = None
        self.refresh_cameras_callback: Optional[Callable] = None
        
        self.running = False
        
        # Переменные GUI
        self._setup_variables()
        self._setup_styles()
        self._create_widgets()
        
    def _setup_variables(self):
        """Инициализация переменных GUI"""
        # Режим работы
        self.mode = tk.StringVar(value="both")
        self.enable_pose = tk.BooleanVar(value=True)
        self.enable_barbell = tk.BooleanVar(value=True)
        
        # Источник видео
        self.source_var = tk.StringVar(value=str(self.camera_list[0]) if self.camera_list else "0")
        
        # Внешний вид
        self.show_joints = tk.BooleanVar(value=True)
        self.bone_color = tk.StringVar(value="#FF6B35")
        self.joint_color = tk.StringVar(value="#4ECDC4")
        self.bone_width = tk.IntVar(value=6)
        self.joint_radius = tk.IntVar(value=6)
        self.font_size = tk.DoubleVar(value=0.7)  # Размер шрифта для градусов
        self.font_thickness = tk.IntVar(value=1)  # Толщина шрифта
        
        # Параметры визуализации пути штанги
        import config
        self.barbell_path_offset_x = tk.IntVar(value=config.BARBELL_PATH_OFFSET_X)
        self.barbell_path_opacity = tk.DoubleVar(value=config.BARBELL_PATH_OPACITY)
        self.barbell_path_color = tk.StringVar(value="#FF0000")  # Красный в HEX
        self.barbell_dash_length = tk.IntVar(value=config.BARBELL_DASH_LENGTH)
        self.barbell_dash_gap = tk.IntVar(value=config.BARBELL_DASH_GAP)
        self.barbell_dash_thickness = tk.IntVar(value=config.BARBELL_DASH_THICKNESS)
        self.barbell_dash_opacity = tk.DoubleVar(value=config.BARBELL_DASH_OPACITY)
        self.barbell_dash_color = tk.StringVar(value="#FFFFFF")  # Белый в HEX
        
        # Модель
        self.model_complexity = tk.IntVar(value=1)
        self.smooth_landmarks = tk.BooleanVar(value=True)
        self.min_det = tk.DoubleVar(value=0.4)
        self.min_track = tk.DoubleVar(value=0.4)
        
        # Выходные потоки
        self.use_ndi = tk.BooleanVar(value=False)
        self.use_virtual = tk.BooleanVar(value=False)
        self.ndi_name = tk.StringVar(value="Stream_NDI")
        
        # Статус
        self.status_var = tk.StringVar(value="Готов к работе")
        
        # Слепые зоны скелета (в долях от 0 до 1, где 0.2 = 20% с каждой стороны)
        self.skeleton_blind_zone_x = tk.DoubleVar(value=0.2)
        self.skeleton_blind_zone_y = tk.DoubleVar(value=0.2)
        
        # Параметры обработки
        self.proc_w = 320
        self.proc_h = 180
        self.every_n = 1
        self.target_fps = 120
        
    def _setup_styles(self):
        """Настройка стилей интерфейса"""
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#2b2b2b')
        self.style.configure('TLabel', background='#2b2b2b', foreground='white')
        self.style.configure('TLabelframe', background='#2b2b2b', foreground='white')
        self.style.configure('TLabelframe.Label', background='#2b2b2b', foreground='white')
        self.style.configure('TButton', background='#404040', foreground='black')  # Изменено на черный
        self.style.configure('TCheckbutton', background='#2b2b2b', foreground='white')
        self.style.configure('TCombobox', background='#404040', foreground='black')  # Изменено на черный
        self.style.configure('TEntry', background='#404040', foreground='black')  # Изменено на черный
        self.style.configure('TScale', background='#2b2b2b')
        self.style.configure('TRadiobutton', background='#2b2b2b', foreground='white')
        
        # Дополнительные настройки для лучшей читаемости
        self.style.map('TButton',
                      foreground=[('pressed', 'black'), ('active', 'black')],
                      background=[('pressed', '!disabled', '#505050'), ('active', '#484848')])
        
        self.style.map('TCombobox',
                      fieldbackground=[('readonly', '#404040')],
                      selectbackground=[('readonly', '#505050')],
                      selectforeground=[('readonly', 'black')])
        
        self.style.map('TEntry',
                      fieldbackground=[('readonly', '#404040')],
                      selectbackground=[('readonly', '#505050')],
                      selectforeground=[('readonly', 'black')])
        
    def _create_widgets(self):
        """Создание всех виджетов интерфейса"""
        self.root.configure(bg='#2b2b2b')
        
        # Основной layout
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Левая панель - настройки с прокруткой
        left_container = ttk.Frame(main_frame, width=400)
        left_container.pack(side='left', fill='both', padx=(0, 10))
        left_container.pack_propagate(False)
        
        # Создаем Canvas и Scrollbar для прокрутки
        self.canvas = tk.Canvas(left_container, bg='#2b2b2b', highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=self.canvas.yview)
        
        # Фрейм для содержимого с прокруткой
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        # Создаем окно в canvas для scrollable_frame
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        # Упаковываем canvas и scrollbar
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Привязываем колесо мыши к прокрутке
        self._bind_mouse_wheel()
        
        # Правая панель - предпросмотр
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side='right', fill='both', expand=True)
        
        self._create_preview_section(right_frame)
        self._create_control_sections(self.scrollable_frame)
        
    def _bind_mouse_wheel(self):
        """Привязка колеса мыши к прокрутке"""
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", self._on_mousewheel)  # Linux
        self.canvas.bind("<Button-5>", self._on_mousewheel)  # Linux
        
    def _on_mousewheel(self, event):
        """Обработчик прокрутки колеса мыши"""
        if event.delta:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        else:
            if event.num == 4:
                self.canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                self.canvas.yview_scroll(1, "units")
        
    def _create_preview_section(self, parent):
        """Создание секции предпросмотра"""
        preview_header = ttk.Label(parent, text="Предпросмотр", font=('Arial', 12, 'bold'))
        preview_header.pack(pady=(0, 5))
        
        preview_container = ttk.Frame(parent)
        preview_container.pack(fill='both', expand=True)
        
        self.preview_label = ttk.Label(
            preview_container, 
            text="Запустите стрим для предпросмотра",
            background='black', 
            foreground='white', 
            font=('Arial', 10), 
            anchor='center'
        )
        self.preview_label.pack(fill='both', expand=True)
        
    def _create_control_sections(self, parent):
        """Создание секций управления"""
        # Режим работы
        self._create_mode_section(parent)
        
        # Источник видео
        self._create_source_section(parent)
        
        # Настройки обработки
        self._create_processing_section(parent)
        
        # Внешний вид
        self._create_appearance_section(parent)
        
        # Настройки шрифта
        self._create_font_section(parent)
        
        # Настройки визуализации пути штанги
        self._create_barbell_path_section(parent)
        
        # Настройки модели
        self._create_model_section(parent)
        
        # Слепые зоны скелета
        self._create_skeleton_blind_zones_section(parent)
        
        # Выходные потоки
        self._create_output_section(parent)
        
        # Управление
        self._create_control_buttons(parent)
        
        # Статус
        self._create_status_section(parent)
        
    def _create_mode_section(self, parent):
        """Секция выбора режима работы"""
        mode_frame = ttk.LabelFrame(parent, text="🎯 Режим работы", padding=10)
        mode_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Radiobutton(
            mode_frame, 
            text="Только поза", 
            variable=self.mode, 
            value="pose",
            command=self.on_mode_change
        ).pack(anchor='w', pady=2)
        
        ttk.Radiobutton(
            mode_frame, 
            text="Только штанга", 
            variable=self.mode, 
            value="barbell",
            command=self.on_mode_change
        ).pack(anchor='w', pady=2)
        
        # ttk.Radiobutton(
        #     mode_frame, 
        #     text="Поза + штанга", 
        #     variable=self.mode, 
        #     value="both",
        #     command=self.on_mode_change
        # ).pack(anchor='w', pady=2)
        
    def _create_source_section(self, parent):
        """Секция выбора источника видео"""
        source_frame = ttk.LabelFrame(parent, text="📷 Источник видео", padding=10)
        source_frame.pack(fill='x', pady=(0, 10))
        
        # Выбор камеры
        source_row1 = ttk.Frame(source_frame)
        source_row1.pack(fill='x', pady=2)
        ttk.Label(source_row1, text="Камера:").pack(side='left')
        
        self.source_combo = ttk.Combobox(
            source_row1, 
            values=[str(x) for x in self.camera_list],
            textvariable=self.source_var, 
            width=12
        )
        self.source_combo.pack(side='left', padx=5)
        
        # Кнопки управления
        source_row2 = ttk.Frame(source_frame)
        source_row2.pack(fill='x', pady=2)
        
        ttk.Button(
            source_row2, 
            text="📁 Выбрать видео", 
            command=self.browse_file
        ).pack(side='left', padx=2)
        
        ttk.Button(
            source_row2, 
            text="🔄 Обновить камеры", 
            command=self.refresh_cameras
        ).pack(side='left', padx=2)
        
        # Отображение выбранного файла
        source_row3 = ttk.Frame(source_frame)
        source_row3.pack(fill='x', pady=2)
        ttk.Label(source_row3, text="Файл:").pack(side='left')
        
        self.file_label = ttk.Label(
            source_row3, 
            text="(не выбран)", 
            foreground='gray', 
            font=('Arial', 8)
        )
        self.file_label.pack(side='left', padx=5)
        
    def _create_processing_section(self, parent):
        """Секция настроек обработки"""
        processing_frame = ttk.LabelFrame(parent, text="⚙️ Настройки обработки", padding=10)
        processing_frame.pack(fill='x', pady=(0, 10))
        
        # Разрешение обработки
        proc_row1 = ttk.Frame(processing_frame)
        proc_row1.pack(fill='x', pady=2)
        ttk.Label(proc_row1, text="Разрешение:").pack(side='left')
        
        self.proc_entry = ttk.Entry(proc_row1, width=10)
        self.proc_entry.insert(0, f"{self.proc_w}x{self.proc_h}")
        self.proc_entry.pack(side='left', padx=5)
        
        # Частота обработки кадров
        ttk.Label(proc_row1, text="Кадры:").pack(side='left', padx=(10,0))
        self.every_spin = ttk.Spinbox(proc_row1, from_=1, to=6, width=4)
        self.every_spin.delete(0, "end")
        self.every_spin.insert(0, str(self.every_n))
        self.every_spin.pack(side='left', padx=5)
        
        # FPS
        proc_row2 = ttk.Frame(processing_frame)
        proc_row2.pack(fill='x', pady=2)
        ttk.Label(proc_row2, text="FPS:").pack(side='left')
        
        self.fps_spin = ttk.Spinbox(proc_row2, from_=5, to=60, width=4)
        self.fps_spin.delete(0, "end")
        self.fps_spin.insert(0, str(self.target_fps))
        self.fps_spin.pack(side='left', padx=5)
        
    def _create_appearance_section(self, parent):
        """Секция настроек внешнего вида"""
        appearance_frame = ttk.LabelFrame(parent, text="🎨 Внешний вид", padding=10)
        appearance_frame.pack(fill='x', pady=(0, 10))
        
        # Цвета
        colors_frame = ttk.Frame(appearance_frame)
        colors_frame.pack(fill='x', pady=5)
        
        # Цвет костей
        ttk.Label(colors_frame, text="Цвет костей:").grid(row=0, column=0, sticky="w", padx=5, pady=3)
        self.bone_color_btn = ttk.Button(
            colors_frame, 
            text="Выбрать", 
            command=self.choose_bone_color, 
            width=8
        )
        self.bone_color_btn.grid(row=0, column=1, padx=5, pady=3)
        
        self.bone_color_preview = tk.Canvas(
            colors_frame, 
            width=40, 
            height=20, 
            bg=self.bone_color.get(), 
            relief='solid', 
            bd=1
        )
        self.bone_color_preview.grid(row=0, column=2, padx=5, pady=3)
        
        # Цвет суставов
        ttk.Label(colors_frame, text="Цвет суставов:").grid(row=1, column=0, sticky="w", padx=5, pady=3)
        self.joint_color_btn = ttk.Button(
            colors_frame, 
            text="Выбрать", 
            command=self.choose_joint_color, 
            width=8
        )
        self.joint_color_btn.grid(row=1, column=1, padx=5, pady=3)
        
        self.joint_color_preview = tk.Canvas(
            colors_frame, 
            width=40, 
            height=20, 
            bg=self.joint_color.get(), 
            relief='solid', 
            bd=1
        )
        self.joint_color_preview.grid(row=1, column=2, padx=5, pady=3)
        
        # Размеры элементов
        sizes_frame = ttk.Frame(appearance_frame)
        sizes_frame.pack(fill='x', pady=5)
        
        # Толщина костей
        ttk.Label(sizes_frame, text="Толщина костей:").grid(row=0, column=0, sticky="w", padx=5, pady=3)
        bone_scale_frame = ttk.Frame(sizes_frame)
        bone_scale_frame.grid(row=0, column=1, columnspan=2, sticky='ew', padx=5, pady=3)
        
        ttk.Scale(
            bone_scale_frame, 
            from_=1, 
            to=20, 
            orient='horizontal', 
            variable=self.bone_width,
            command=self.on_bone_width_change, 
            length=120
        ).pack(side='left')
        
        self.bone_width_label = ttk.Label(
            bone_scale_frame, 
            text=str(self.bone_width.get()), 
            width=3
        )
        self.bone_width_label.pack(side='left', padx=5)
        
        # Размер суставов
        ttk.Label(sizes_frame, text="Размер суставов:").grid(row=1, column=0, sticky="w", padx=5, pady=3)
        joint_scale_frame = ttk.Frame(sizes_frame)
        joint_scale_frame.grid(row=1, column=1, columnspan=2, sticky='ew', padx=5, pady=3)
        
        ttk.Scale(
            joint_scale_frame, 
            from_=1, 
            to=20, 
            orient='horizontal', 
            variable=self.joint_radius,
            command=self.on_joint_radius_change, 
            length=120
        ).pack(side='left')
        
        self.joint_radius_label = ttk.Label(
            joint_scale_frame, 
            text=str(self.joint_radius.get()), 
            width=3
        )
        self.joint_radius_label.pack(side='left', padx=5)
        
    def _create_font_section(self, parent):
        """Секция настроек шрифта для градусов"""
        font_frame = ttk.LabelFrame(parent, text="🔤 Настройки шрифта", padding=10)
        font_frame.pack(fill='x', pady=(0, 10))
        
        # Размер шрифта
        font_size_frame = ttk.Frame(font_frame)
        font_size_frame.pack(fill='x', pady=5)
        
        ttk.Label(font_size_frame, text="Размер шрифта:").grid(row=0, column=0, sticky="w", padx=5, pady=3)
        font_size_scale_frame = ttk.Frame(font_size_frame)
        font_size_scale_frame.grid(row=0, column=1, columnspan=2, sticky='ew', padx=5, pady=3)
        
        ttk.Scale(
            font_size_scale_frame, 
            from_=0.3, 
            to=2.0, 
            orient='horizontal', 
            variable=self.font_size,
            command=self.on_font_size_change, 
            length=120
        ).pack(side='left')
        
        self.font_size_label = ttk.Label(
            font_size_scale_frame, 
            text=f"{self.font_size.get():.1f}", 
            width=3
        )
        self.font_size_label.pack(side='left', padx=5)
        
        # Толщина шрифта
        font_thickness_frame = ttk.Frame(font_frame)
        font_thickness_frame.pack(fill='x', pady=5)
        
        ttk.Label(font_thickness_frame, text="Толщина шрифта:").grid(row=0, column=0, sticky="w", padx=5, pady=3)
        font_thickness_scale_frame = ttk.Frame(font_thickness_frame)
        font_thickness_scale_frame.grid(row=0, column=1, columnspan=2, sticky='ew', padx=5, pady=3)
        
        ttk.Scale(
            font_thickness_scale_frame, 
            from_=1, 
            to=5, 
            orient='horizontal', 
            variable=self.font_thickness,
            command=self.on_font_thickness_change, 
            length=120
        ).pack(side='left')
        
        self.font_thickness_label = ttk.Label(
            font_thickness_scale_frame, 
            text=str(self.font_thickness.get()), 
            width=3
        )
        self.font_thickness_label.pack(side='left', padx=5)
    
    def _create_barbell_path_section(self, parent):
        """Секция настроек визуализации пути штанги"""
        barbell_frame = ttk.LabelFrame(parent, text="🎯 Визуализация пути штанги", padding=10)
        barbell_frame.pack(fill='x', pady=(0, 10))
        
        import config
        
        # Смещение пути вправо
        offset_frame = ttk.Frame(barbell_frame)
        offset_frame.pack(fill='x', pady=5)
        ttk.Label(offset_frame, text="Смещение пути (X):").grid(row=0, column=0, sticky="w", padx=5, pady=3)
        offset_scale_frame = ttk.Frame(offset_frame)
        offset_scale_frame.grid(row=0, column=1, columnspan=2, sticky='ew', padx=5, pady=3)
        
        ttk.Scale(
            offset_scale_frame,
            from_=0,
            to=1000,
            orient='horizontal',
            variable=self.barbell_path_offset_x,
            command=self.on_barbell_path_offset_change,
            length=120
        ).pack(side='left')
        
        self.barbell_path_offset_label = ttk.Label(
            offset_scale_frame,
            text=str(self.barbell_path_offset_x.get()),
            width=4
        )
        self.barbell_path_offset_label.pack(side='left', padx=5)
        
        # Длина сегмента пунктира
        dash_length_frame = ttk.Frame(barbell_frame)
        dash_length_frame.pack(fill='x', pady=5)
        ttk.Label(dash_length_frame, text="Длина сегмента пунктира:").grid(row=0, column=0, sticky="w", padx=5, pady=3)
        dash_length_scale_frame = ttk.Frame(dash_length_frame)
        dash_length_scale_frame.grid(row=0, column=1, columnspan=2, sticky='ew', padx=5, pady=3)
        
        ttk.Scale(
            dash_length_scale_frame,
            from_=1,
            to=50,
            orient='horizontal',
            variable=self.barbell_dash_length,
            command=self.on_barbell_dash_length_change,
            length=120
        ).pack(side='left')
        
        self.barbell_dash_length_label = ttk.Label(
            dash_length_scale_frame,
            text=str(self.barbell_dash_length.get()),
            width=3
        )
        self.barbell_dash_length_label.pack(side='left', padx=5)
        
        # Промежуток между сегментами пунктира
        dash_gap_frame = ttk.Frame(barbell_frame)
        dash_gap_frame.pack(fill='x', pady=5)
        ttk.Label(dash_gap_frame, text="Промежуток пунктира:").grid(row=0, column=0, sticky="w", padx=5, pady=3)
        dash_gap_scale_frame = ttk.Frame(dash_gap_frame)
        dash_gap_scale_frame.grid(row=0, column=1, columnspan=2, sticky='ew', padx=5, pady=3)
        
        ttk.Scale(
            dash_gap_scale_frame,
            from_=1,
            to=50,
            orient='horizontal',
            variable=self.barbell_dash_gap,
            command=self.on_barbell_dash_gap_change,
            length=120
        ).pack(side='left')
        
        self.barbell_dash_gap_label = ttk.Label(
            dash_gap_scale_frame,
            text=str(self.barbell_dash_gap.get()),
            width=3
        )
        self.barbell_dash_gap_label.pack(side='left', padx=5)
        
        # Толщина пунктира
        dash_thickness_frame = ttk.Frame(barbell_frame)
        dash_thickness_frame.pack(fill='x', pady=5)
        ttk.Label(dash_thickness_frame, text="Толщина пунктира:").grid(row=0, column=0, sticky="w", padx=5, pady=3)
        dash_thickness_scale_frame = ttk.Frame(dash_thickness_frame)
        dash_thickness_scale_frame.grid(row=0, column=1, columnspan=2, sticky='ew', padx=5, pady=3)
        
        ttk.Scale(
            dash_thickness_scale_frame,
            from_=1,
            to=10,
            orient='horizontal',
            variable=self.barbell_dash_thickness,
            command=self.on_barbell_dash_thickness_change,
            length=120
        ).pack(side='left')
        
        self.barbell_dash_thickness_label = ttk.Label(
            dash_thickness_scale_frame,
            text=str(self.barbell_dash_thickness.get()),
            width=3
        )
        self.barbell_dash_thickness_label.pack(side='left', padx=5)
        
        # Прозрачность пунктира
        dash_opacity_frame = ttk.Frame(barbell_frame)
        dash_opacity_frame.pack(fill='x', pady=5)
        ttk.Label(dash_opacity_frame, text="Прозрачность пунктира:").grid(row=0, column=0, sticky="w", padx=5, pady=3)
        dash_opacity_scale_frame = ttk.Frame(dash_opacity_frame)
        dash_opacity_scale_frame.grid(row=0, column=1, columnspan=2, sticky='ew', padx=5, pady=3)
        
        ttk.Scale(
            dash_opacity_scale_frame,
            from_=0.0,
            to=1.0,
            orient='horizontal',
            variable=self.barbell_dash_opacity,
            command=self.on_barbell_dash_opacity_change,
            length=120
        ).pack(side='left')
        
        self.barbell_dash_opacity_label = ttk.Label(
            dash_opacity_scale_frame,
            text=f"{self.barbell_dash_opacity.get():.2f}",
            width=4
        )
        self.barbell_dash_opacity_label.pack(side='left', padx=5)
        
        # Цвет пути
        path_color_frame = ttk.Frame(barbell_frame)
        path_color_frame.pack(fill='x', pady=5)
        ttk.Label(path_color_frame, text="Цвет пути:").grid(row=0, column=0, sticky="w", padx=5, pady=3)
        self.barbell_path_color_btn = ttk.Button(
            path_color_frame,
            text="Выбрать",
            command=self.choose_barbell_path_color,
            width=8
        )
        self.barbell_path_color_btn.grid(row=0, column=1, padx=5, pady=3)
        self.barbell_path_color_preview = tk.Canvas(
            path_color_frame,
            width=40,
            height=20,
            bg=self.barbell_path_color.get(),
            relief='solid',
            bd=1
        )
        self.barbell_path_color_preview.grid(row=0, column=2, padx=5, pady=3)
        
        # Цвет пунктира
        dash_color_frame = ttk.Frame(barbell_frame)
        dash_color_frame.pack(fill='x', pady=5)
        ttk.Label(dash_color_frame, text="Цвет пунктира:").grid(row=0, column=0, sticky="w", padx=5, pady=3)
        self.barbell_dash_color_btn = ttk.Button(
            dash_color_frame,
            text="Выбрать",
            command=self.choose_barbell_dash_color,
            width=8
        )
        self.barbell_dash_color_btn.grid(row=0, column=1, padx=5, pady=3)
        self.barbell_dash_color_preview = tk.Canvas(
            dash_color_frame,
            width=40,
            height=20,
            bg=self.barbell_dash_color.get(),
            relief='solid',
            bd=1
        )
        self.barbell_dash_color_preview.grid(row=0, column=2, padx=5, pady=3)
        
        # Прозрачность пути
        path_opacity_frame = ttk.Frame(barbell_frame)
        path_opacity_frame.pack(fill='x', pady=5)
        ttk.Label(path_opacity_frame, text="Прозрачность пути:").grid(row=0, column=0, sticky="w", padx=5, pady=3)
        path_opacity_scale_frame = ttk.Frame(path_opacity_frame)
        path_opacity_scale_frame.grid(row=0, column=1, columnspan=2, sticky='ew', padx=5, pady=3)
        
        ttk.Scale(
            path_opacity_scale_frame,
            from_=0.0,
            to=1.0,
            orient='horizontal',
            variable=self.barbell_path_opacity,
            command=self.on_barbell_path_opacity_change,
            length=120
        ).pack(side='left')
        
        self.barbell_path_opacity_label = ttk.Label(
            path_opacity_scale_frame,
            text=f"{self.barbell_path_opacity.get():.2f}",
            width=4
        )
        self.barbell_path_opacity_label.pack(side='left', padx=5)
    
    def _create_model_section(self, parent):
        """Секция настроек модели"""
        model_frame = ttk.LabelFrame(parent, text="🧠 Настройки модели", padding=10)
        model_frame.pack(fill='x', pady=(0, 10))
        
        # Сложность модели
        model_row1 = ttk.Frame(model_frame)
        model_row1.pack(fill='x', pady=2)
        ttk.Label(model_row1, text="Сложность:").pack(side='left')
        
        ttk.Spinbox(
            model_row1, 
            from_=0, 
            to=1, 
            width=5, 
            textvariable=self.model_complexity
        ).pack(side='left', padx=5)
        
        ttk.Checkbutton(
            model_row1, 
            text="Сглаживание", 
            variable=self.smooth_landmarks
        ).pack(side='left', padx=10)
        
        # Пороги детекции и трекинга
        model_row2 = ttk.Frame(model_frame)
        model_row2.pack(fill='x', pady=2)
        ttk.Label(model_row2, text="Детекция:").pack(side='left')
        
        ttk.Entry(model_row2, textvariable=self.min_det, width=6).pack(side='left', padx=5)
        
        ttk.Label(model_row2, text="Трекинг:").pack(side='left', padx=(10,0))
        ttk.Entry(model_row2, textvariable=self.min_track, width=6).pack(side='left', padx=5)
    
    def _create_skeleton_blind_zones_section(self, parent):
        """Секция настроек слепых зон скелета"""
        blind_zones_frame = ttk.LabelFrame(parent, text="👁️ Слепые зоны скелета", padding=10)
        blind_zones_frame.pack(fill='x', pady=(0, 10))
        
        # Слепая зона по X
        blind_zone_x_frame = ttk.Frame(blind_zones_frame)
        blind_zone_x_frame.pack(fill='x', pady=5)
        ttk.Label(blind_zone_x_frame, text="Слепая зона по X (%):").grid(row=0, column=0, sticky="w", padx=5, pady=3)
        blind_zone_x_scale_frame = ttk.Frame(blind_zone_x_frame)
        blind_zone_x_scale_frame.grid(row=0, column=1, columnspan=2, sticky='ew', padx=5, pady=3)
        
        ttk.Scale(
            blind_zone_x_scale_frame,
            from_=0.0,
            to=0.5,
            orient='horizontal',
            variable=self.skeleton_blind_zone_x,
            command=self.on_blind_zone_x_change,
            length=120
        ).pack(side='left')
        
        self.blind_zone_x_label = ttk.Label(
            blind_zone_x_scale_frame,
            text=f"{self.skeleton_blind_zone_x.get()*100:.1f}%",
            width=6
        )
        self.blind_zone_x_label.pack(side='left', padx=5)
        
        # Слепая зона по Y
        blind_zone_y_frame = ttk.Frame(blind_zones_frame)
        blind_zone_y_frame.pack(fill='x', pady=5)
        ttk.Label(blind_zone_y_frame, text="Слепая зона по Y (%):").grid(row=0, column=0, sticky="w", padx=5, pady=3)
        blind_zone_y_scale_frame = ttk.Frame(blind_zone_y_frame)
        blind_zone_y_scale_frame.grid(row=0, column=1, columnspan=2, sticky='ew', padx=5, pady=3)
        
        ttk.Scale(
            blind_zone_y_scale_frame,
            from_=0.0,
            to=0.5,
            orient='horizontal',
            variable=self.skeleton_blind_zone_y,
            command=self.on_blind_zone_y_change,
            length=120
        ).pack(side='left')
        
        self.blind_zone_y_label = ttk.Label(
            blind_zone_y_scale_frame,
            text=f"{self.skeleton_blind_zone_y.get()*100:.1f}%",
            width=6
        )
        self.blind_zone_y_label.pack(side='left', padx=5)
        
        # Информационная подсказка
        info_label = ttk.Label(
            blind_zones_frame,
            text="Скелет скрывается, если человек выходит за пределы центральной зоны",
            font=('Arial', 8),
            foreground='gray'
        )
        info_label.pack(pady=(5, 0))
    
    def _create_output_section(self, parent):
        """Секция выходных потоков"""
        output_frame = ttk.LabelFrame(parent, text="📤 Выходные потоки", padding=10)
        output_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Checkbutton(
            output_frame, 
            text="Показывать скелет", 
            variable=self.show_joints
        ).pack(anchor='w', pady=2)
        
        ttk.Checkbutton(
            output_frame, 
            text="Использовать NDI", 
            variable=self.use_ndi
        ).pack(anchor='w', pady=2)
        
        ttk.Checkbutton(
            output_frame, 
            text="Виртуальная камера", 
            variable=self.use_virtual
        ).pack(anchor='w', pady=2)
        
        # Настройка NDI
        ndi_frame = ttk.Frame(output_frame)
        ndi_frame.pack(fill='x', pady=2)
        ttk.Label(ndi_frame, text="Имя NDI:").pack(side='left')
        
        ttk.Entry(ndi_frame, textvariable=self.ndi_name, width=15).pack(side='left', padx=5)
        
    def _create_control_buttons(self, parent):
        """Секция кнопок управления"""
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill='x', pady=10)
        
        # Основные кнопки управления
        main_buttons_frame = ttk.Frame(control_frame)
        main_buttons_frame.pack(fill='x', pady=(0, 5))
        
        self.start_btn = ttk.Button(
            main_buttons_frame, 
            text="▶️ Запуск", 
            command=self.start, 
            width=12
        )
        self.start_btn.pack(side='left', padx=2)
        
        self.stop_btn = ttk.Button(
            main_buttons_frame, 
            text="⏹️ Остановка", 
            command=self.stop, 
            state="disabled", 
            width=12
        )
        self.stop_btn.pack(side='left', padx=2)
        
        ttk.Button(
            main_buttons_frame, 
            text="❌ Выход", 
            command=self.quit, 
            width=12
        ).pack(side='left', padx=2)
        
        # Кнопки сохранения/загрузки настроек
        settings_buttons_frame = ttk.Frame(control_frame)
        settings_buttons_frame.pack(fill='x')
        
        ttk.Button(
            settings_buttons_frame,
            text="💾 Сохранить настройки",
            command=self.save_settings,
            width=18
        ).pack(side='left', padx=2)
        
        ttk.Button(
            settings_buttons_frame,
            text="📂 Загрузить настройки",
            command=self.load_settings,
            width=18
        ).pack(side='left', padx=2)
        
    def _create_status_section(self, parent):
        """Секция статуса"""
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill='x', pady=5)
        
        status_label = ttk.Label(
            status_frame, 
            textvariable=self.status_var, 
            relief="sunken",
            anchor="center", 
            background='#404040', 
            foreground='white'
        )
        status_label.pack(fill='x')
        
    # === PUBLIC METHODS ===
    
    def set_start_callback(self, callback: Callable):
        """Установка callback для запуска"""
        self.start_callback = callback
        
    def set_stop_callback(self, callback: Callable):
        """Установка callback для остановки"""
        self.stop_callback = callback
        
    def set_quit_callback(self, callback: Callable):
        """Установка callback для выхода"""
        self.quit_callback = callback
        
    def set_refresh_cameras_callback(self, callback: Callable):
        """Установка callback для обновления камер"""
        self.refresh_cameras_callback = callback
        
    def update_preview(self, frame):
        """Обновление изображения предпросмотра"""
        try:
            import cv2
            preview_frame = cv2.resize(frame, (640, 360))
            preview_frame = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
            
            from PIL import Image, ImageTk
            import io
            
            # Конвертируем frame в PhotoImage
            image = Image.fromarray(preview_frame)
            photo = ImageTk.PhotoImage(image=image)
            
            self.preview_label.configure(image=photo)
            self.preview_label.image = photo  # Сохраняем ссылку
            
        except Exception as e:
            print(f"Preview update error: {e}")
            
    def update_status(self, status: str):
        """Обновление статуса"""
        self.status_var.set(status)
        
    def set_running_state(self, running: bool):
        """Установка состояния работы"""
        self.running = running
        if running:
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
        else:
            self.start_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            
    def update_camera_list(self, cameras: list):
        """Обновление списка камер"""
        self.camera_list = cameras
        current_value = self.source_var.get()
        
        # Сохраняем текущее значение если это файл
        if current_value and (current_value.lower().endswith((".mp4", ".mov", ".avi")) or 
                             (current_value.isdigit() and int(current_value) in cameras)):
            cam_values = [str(x) for x in cameras]
            if current_value not in cam_values:
                cam_values.append(current_value)
            self.source_combo['values'] = cam_values
        else:
            self.source_combo['values'] = [str(x) for x in cameras]
            if cameras:
                self.source_var.set(str(cameras[0]))
                
    def get_processing_params(self):
        """Получение параметров обработки"""
        try:
            proc_res = self.proc_entry.get().strip()
            pw, ph = [int(x) for x in proc_res.split("x")]
            every_n = max(1, int(self.every_spin.get()))
            target_fps = int(self.fps_spin.get())
            return pw, ph, every_n, target_fps
        except Exception as e:
            raise ValueError(f"Ошибка в параметрах обработки: {e}")
            
    def get_source(self):
        """Получение выбранного источника"""
        return self.source_var.get()
        
    def get_font_settings(self):
        """Получение настроек шрифта"""
        return {
            'font_size': self.font_size.get(),
            'font_thickness': self.font_thickness.get()
        }
        
    # === EVENT HANDLERS ===
    
    def on_mode_change(self):
        """Обработчик изменения режима"""
        mode = self.mode.get()
        self.enable_pose.set(mode == "pose" or mode == "both")
        self.enable_barbell.set(mode == "barbell" or mode == "both")
        
    def choose_bone_color(self):
        """Выбор цвета костей"""
        color = askcolor(initialcolor=self.bone_color.get(), title="Выберите цвет костей")[1]
        if color:
            self.bone_color.set(color)
            self.bone_color_preview.config(bg=color)
            
    def choose_joint_color(self):
        """Выбор цвета суставов"""
        color = askcolor(initialcolor=self.joint_color.get(), title="Выберите цвет суставов")[1]
        if color:
            self.joint_color.set(color)
            self.joint_color_preview.config(bg=color)
            
    def on_bone_width_change(self, value):
        """Обработчик изменения толщины костей"""
        self.bone_width_label.config(text=str(int(float(value))))
        
    def on_joint_radius_change(self, value):
        """Обработчик изменения размера суставов"""
        self.joint_radius_label.config(text=str(int(float(value))))
        
    def on_font_size_change(self, value):
        """Обработчик изменения размера шрифта"""
        self.font_size_label.config(text=f"{float(value):.1f}")
        
    def on_font_thickness_change(self, value):
        """Обработчик изменения толщины шрифта"""
        self.font_thickness_label.config(text=str(int(float(value))))
    
    def on_barbell_path_offset_change(self, value):
        """Обработчик изменения смещения пути"""
        val = int(float(value))
        self.barbell_path_offset_label.config(text=str(val))
        import config
        config.BARBELL_PATH_OFFSET_X = val
    
    def on_barbell_dash_length_change(self, value):
        """Обработчик изменения длины сегмента пунктира"""
        val = int(float(value))
        self.barbell_dash_length_label.config(text=str(val))
        import config
        config.BARBELL_DASH_LENGTH = val
    
    def on_barbell_dash_gap_change(self, value):
        """Обработчик изменения промежутка пунктира"""
        val = int(float(value))
        self.barbell_dash_gap_label.config(text=str(val))
        import config
        config.BARBELL_DASH_GAP = val
    
    def on_barbell_dash_thickness_change(self, value):
        """Обработчик изменения толщины пунктира"""
        val = int(float(value))
        self.barbell_dash_thickness_label.config(text=str(val))
        import config
        config.BARBELL_DASH_THICKNESS = val
    
    def on_barbell_dash_opacity_change(self, value):
        """Обработчик изменения прозрачности пунктира"""
        val = float(value)
        self.barbell_dash_opacity_label.config(text=f"{val:.2f}")
        import config
        config.BARBELL_DASH_OPACITY = val
    
    def on_barbell_path_opacity_change(self, value):
        """Обработчик изменения прозрачности пути"""
        val = float(value)
        self.barbell_path_opacity_label.config(text=f"{val:.2f}")
        import config
        config.BARBELL_PATH_OPACITY = val
    
    def choose_barbell_path_color(self):
        """Выбор цвета пути"""
        color = askcolor(initialcolor=self.barbell_path_color.get(), title="Выберите цвет пути")[1]
        if color:
            self.barbell_path_color.set(color)
            self.barbell_path_color_preview.config(bg=color)
            # Конвертируем HEX в BGR для config
            import config
            rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
            config.BARBELL_PATH_COLOR = (rgb[2], rgb[1], rgb[0])  # RGB -> BGR
    
    def choose_barbell_dash_color(self):
        """Выбор цвета пунктира"""
        color = askcolor(initialcolor=self.barbell_dash_color.get(), title="Выберите цвет пунктира")[1]
        if color:
            self.barbell_dash_color.set(color)
            self.barbell_dash_color_preview.config(bg=color)
            # Конвертируем HEX в BGR для config
            import config
            rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
            config.BARBELL_DASH_COLOR = (rgb[2], rgb[1], rgb[0])  # RGB -> BGR
    
    def on_blind_zone_x_change(self, value):
        """Обработчик изменения слепой зоны по X"""
        val = float(value)
        self.blind_zone_x_label.config(text=f"{val*100:.1f}%")
    
    def on_blind_zone_y_change(self, value):
        """Обработчик изменения слепой зоны по Y"""
        val = float(value)
        self.blind_zone_y_label.config(text=f"{val*100:.1f}%")
    
    def get_blind_zones(self):
        """Получение значений слепых зон"""
        return {
            'blind_zone_x': self.skeleton_blind_zone_x.get(),
            'blind_zone_y': self.skeleton_blind_zone_y.get()
        }
    
    def save_settings(self, filepath=None):
        """Сохранение текущих настроек в файл"""
        
        if filepath is None:
            filepath = filedialog.asksaveasfilename(
                title="Сохранить настройки",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if not filepath:
                return False
        
        try:
            settings = {
                # Режим работы
                'mode': self.mode.get(),
                'enable_pose': self.enable_pose.get(),
                'enable_barbell': self.enable_barbell.get(),
                
                # Источник видео
                'source': self.source_var.get(),
                
                # Внешний вид
                'show_joints': self.show_joints.get(),
                'bone_color': self.bone_color.get(),
                'joint_color': self.joint_color.get(),
                'bone_width': self.bone_width.get(),
                'joint_radius': self.joint_radius.get(),
                'font_size': self.font_size.get(),
                'font_thickness': self.font_thickness.get(),
                
                # Параметры визуализации пути штанги
                'barbell_path_offset_x': self.barbell_path_offset_x.get(),
                'barbell_path_opacity': self.barbell_path_opacity.get(),
                'barbell_path_color': self.barbell_path_color.get(),
                'barbell_dash_length': self.barbell_dash_length.get(),
                'barbell_dash_gap': self.barbell_dash_gap.get(),
                'barbell_dash_thickness': self.barbell_dash_thickness.get(),
                'barbell_dash_opacity': self.barbell_dash_opacity.get(),
                'barbell_dash_color': self.barbell_dash_color.get(),
                
                # Модель
                'model_complexity': self.model_complexity.get(),
                'smooth_landmarks': self.smooth_landmarks.get(),
                'min_det': self.min_det.get(),
                'min_track': self.min_track.get(),
                
                # Выходные потоки
                'use_ndi': self.use_ndi.get(),
                'use_virtual': self.use_virtual.get(),
                'ndi_name': self.ndi_name.get(),
                
                # Слепые зоны скелета
                'skeleton_blind_zone_x': self.skeleton_blind_zone_x.get(),
                'skeleton_blind_zone_y': self.skeleton_blind_zone_y.get(),
                
                # Параметры обработки
                'proc_w': self.proc_w,
                'proc_h': self.proc_h,
                'every_n': self.every_n,
                'target_fps': self.target_fps
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
            
            messagebox.showinfo("Успех", f"Настройки сохранены в:\n{filepath}")
            return True
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить настройки:\n{e}")
            return False
    
    def load_settings(self, filepath=None):
        """Загрузка настроек из файла"""
        
        if filepath is None:
            filepath = filedialog.askopenfilename(
                title="Загрузить настройки",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if not filepath:
                return False
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                settings = json.load(f)
            
            # Загружаем настройки
            if 'mode' in settings:
                self.mode.set(settings.get('mode', 'both'))
            if 'enable_pose' in settings:
                self.enable_pose.set(settings.get('enable_pose', True))
            if 'enable_barbell' in settings:
                self.enable_barbell.set(settings.get('enable_barbell', True))
            
            if 'source' in settings:
                self.source_var.set(settings['source'])
            
            if 'show_joints' in settings:
                self.show_joints.set(settings.get('show_joints', True))
            if 'bone_color' in settings:
                self.bone_color.set(settings['bone_color'])
                self.bone_color_preview.config(bg=settings['bone_color'])
            if 'joint_color' in settings:
                self.joint_color.set(settings['joint_color'])
                self.joint_color_preview.config(bg=settings['joint_color'])
            if 'bone_width' in settings:
                self.bone_width.set(settings['bone_width'])
                self.bone_width_label.config(text=str(settings['bone_width']))
            if 'joint_radius' in settings:
                self.joint_radius.set(settings['joint_radius'])
                self.joint_radius_label.config(text=str(settings['joint_radius']))
            if 'font_size' in settings:
                self.font_size.set(settings['font_size'])
                self.font_size_label.config(text=f"{settings['font_size']:.1f}")
            if 'font_thickness' in settings:
                self.font_thickness.set(settings['font_thickness'])
                self.font_thickness_label.config(text=str(settings['font_thickness']))
            
            # Параметры визуализации пути штанги
            if 'barbell_path_offset_x' in settings:
                self.barbell_path_offset_x.set(settings['barbell_path_offset_x'])
                self.barbell_path_offset_label.config(text=str(settings['barbell_path_offset_x']))
            if 'barbell_path_opacity' in settings:
                self.barbell_path_opacity.set(settings['barbell_path_opacity'])
                self.barbell_path_opacity_label.config(text=f"{settings['barbell_path_opacity']:.2f}")
            if 'barbell_path_color' in settings:
                self.barbell_path_color.set(settings['barbell_path_color'])
                self.barbell_path_color_preview.config(bg=settings['barbell_path_color'])
            if 'barbell_dash_length' in settings:
                self.barbell_dash_length.set(settings['barbell_dash_length'])
                self.barbell_dash_length_label.config(text=str(settings['barbell_dash_length']))
            if 'barbell_dash_gap' in settings:
                self.barbell_dash_gap.set(settings['barbell_dash_gap'])
                self.barbell_dash_gap_label.config(text=str(settings['barbell_dash_gap']))
            if 'barbell_dash_thickness' in settings:
                self.barbell_dash_thickness.set(settings['barbell_dash_thickness'])
                self.barbell_dash_thickness_label.config(text=str(settings['barbell_dash_thickness']))
            if 'barbell_dash_opacity' in settings:
                self.barbell_dash_opacity.set(settings['barbell_dash_opacity'])
                self.barbell_dash_opacity_label.config(text=f"{settings['barbell_dash_opacity']:.2f}")
            if 'barbell_dash_color' in settings:
                self.barbell_dash_color.set(settings['barbell_dash_color'])
                self.barbell_dash_color_preview.config(bg=settings['barbell_dash_color'])
            
            # Модель
            if 'model_complexity' in settings:
                self.model_complexity.set(settings.get('model_complexity', 1))
            if 'smooth_landmarks' in settings:
                self.smooth_landmarks.set(settings.get('smooth_landmarks', True))
            if 'min_det' in settings:
                self.min_det.set(settings.get('min_det', 0.4))
            if 'min_track' in settings:
                self.min_track.set(settings.get('min_track', 0.4))
            
            # Выходные потоки
            if 'use_ndi' in settings:
                self.use_ndi.set(settings.get('use_ndi', False))
            if 'use_virtual' in settings:
                self.use_virtual.set(settings.get('use_virtual', False))
            if 'ndi_name' in settings:
                self.ndi_name.set(settings.get('ndi_name', 'Stream_NDI'))
            
            # Слепые зоны скелета
            if 'skeleton_blind_zone_x' in settings:
                self.skeleton_blind_zone_x.set(settings['skeleton_blind_zone_x'])
                self.blind_zone_x_label.config(text=f"{settings['skeleton_blind_zone_x']*100:.1f}%")
            if 'skeleton_blind_zone_y' in settings:
                self.skeleton_blind_zone_y.set(settings['skeleton_blind_zone_y'])
                self.blind_zone_y_label.config(text=f"{settings['skeleton_blind_zone_y']*100:.1f}%")
            
            # Параметры обработки
            if 'proc_w' in settings:
                self.proc_w = settings['proc_w']
                self.proc_entry.delete(0, "end")
                self.proc_entry.insert(0, f"{self.proc_w}x{self.proc_h}")
            if 'proc_h' in settings:
                self.proc_h = settings['proc_h']
                self.proc_entry.delete(0, "end")
                self.proc_entry.insert(0, f"{self.proc_w}x{self.proc_h}")
            if 'every_n' in settings:
                self.every_n = settings['every_n']
                self.every_spin.delete(0, "end")
                self.every_spin.insert(0, str(self.every_n))
            if 'target_fps' in settings:
                self.target_fps = settings['target_fps']
                self.fps_spin.delete(0, "end")
                self.fps_spin.insert(0, str(self.target_fps))
            
            # Обновляем config для параметров штанги
            import config
            if 'barbell_path_offset_x' in settings:
                config.BARBELL_PATH_OFFSET_X = settings['barbell_path_offset_x']
            if 'barbell_path_opacity' in settings:
                config.BARBELL_PATH_OPACITY = settings['barbell_path_opacity']
            if 'barbell_path_color' in settings:
                color = settings['barbell_path_color']
                rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
                config.BARBELL_PATH_COLOR = (rgb[2], rgb[1], rgb[0])
            if 'barbell_dash_length' in settings:
                config.BARBELL_DASH_LENGTH = settings['barbell_dash_length']
            if 'barbell_dash_gap' in settings:
                config.BARBELL_DASH_GAP = settings['barbell_dash_gap']
            if 'barbell_dash_thickness' in settings:
                config.BARBELL_DASH_THICKNESS = settings['barbell_dash_thickness']
            if 'barbell_dash_opacity' in settings:
                config.BARBELL_DASH_OPACITY = settings['barbell_dash_opacity']
            if 'barbell_dash_color' in settings:
                color = settings['barbell_dash_color']
                rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
                config.BARBELL_DASH_COLOR = (rgb[2], rgb[1], rgb[0])
            
            messagebox.showinfo("Успех", f"Настройки загружены из:\n{filepath}")
            return True
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить настройки:\n{e}")
            return False
        
    def browse_file(self):
        """Выбор видео файла"""
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
            filename = os.path.basename(path)
            if len(filename) > 30:
                filename = "..." + filename[-27:]
            self.file_label.config(text=filename, foreground='white')
            
            # Добавляем путь в список комбобокса
            current_values = list(self.source_combo['values'])
            if path not in current_values:
                self.source_combo['values'] = current_values + [path]
                
    def refresh_cameras(self):
        """Обновление списка камер"""
        if self.refresh_cameras_callback:
            self.refresh_cameras_callback()
            
    def start(self):
        """Запуск обработки"""
        if self.running:
            return
            
        if self.start_callback:
            try:
                self.start_callback()
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось запустить: {e}")
                
    def stop(self):
        """Остановка обработки"""
        if not self.running:
            return
            
        if self.stop_callback:
            self.stop_callback()
            
    def quit(self):
        """Выход из приложения"""
        if self.quit_callback:
            self.quit_callback()