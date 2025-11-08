"""
Модуль отправки данных через UDP в формате JSON
"""

import socket
import json
import logging
from typing import Optional, Dict, List, Tuple
import time

logger = logging.getLogger(__name__)


class UDPSender:
    """Класс для отправки данных отслеживания через UDP"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 5005):
        """
        Инициализация UDP отправителя
        
        Args:
            host: IP адрес получателя
            port: Порт для отправки
        """
        self.host = host
        self.port = port
        self.socket = None
        self._initialize_socket()
    
    def _initialize_socket(self):
        """Инициализация UDP сокета"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            logger.info(f"UDP сокет инициализирован для {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Ошибка инициализации UDP сокета: {e}")
            self.socket = None
    
    def send_data(self, 
                  knee_positions: Dict,
                  barbell_path: List[Tuple[float, float, float]],
                  timestamp: float):
        """
        Отправка данных отслеживания через UDP
        
        Args:
            knee_positions: Словарь с координатами коленей и углами
            barbell_path: Путь штанги [(x, y, timestamp), ...]
            timestamp: Временная метка текущего кадра
        """
        if not self.socket:
            logger.warning("UDP сокет не инициализирован, данные не отправлены")
            return
        
        # Формирование JSON данных
        # Конвертация координат коленей в список [x, y] или null
        left_knee = knee_positions.get("left_knee")
        right_knee = knee_positions.get("right_knee")
        all_joints = knee_positions.get("all_joints")
        
        data = {
            "timestamp": timestamp,
            "knee_positions": {
                "left_knee": list(left_knee) if left_knee else None,
                "right_knee": list(right_knee) if right_knee else None,
                "left_knee_angle": knee_positions.get("left_knee_angle"),
                "right_knee_angle": knee_positions.get("right_knee_angle")
            },
            "joints": all_joints,  # Все joints для Unreal Engine
            "barbell_path": [
                {"x": float(x), "y": float(y), "timestamp": float(ts)}
                for x, y, ts in barbell_path
            ]
        }
        
        # Конвертация в JSON
        try:
            json_data = json.dumps(data, ensure_ascii=False)
            json_bytes = json_data.encode('utf-8')
            
            # Отправка через UDP
            self.socket.sendto(json_bytes, (self.host, self.port))
            
            logger.debug(f"Данные отправлены через UDP: {len(json_bytes)} байт")
            
        except Exception as e:
            logger.error(f"Ошибка отправки данных через UDP: {e}")
    
    def close(self):
        """Закрытие UDP сокета"""
        if self.socket:
            self.socket.close()
            self.socket = None
            logger.info("UDP сокет закрыт")

