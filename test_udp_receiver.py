"""
Тестовый скрипт для получения данных через UDP
Можно использовать для проверки работы модуля отслеживания
"""

import socket
import json
import sys

def main():
    # Параметры UDP
    host = "127.0.0.1"
    port = 5005
    
    # Создание UDP сокета
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    sock.settimeout(1.0)  # Таймаут 1 секунда
    
    print(f"Ожидание UDP данных на {host}:{port}...")
    print("Нажмите Ctrl+C для выхода\n")
    
    try:
        frame_count = 0
        while True:
            try:
                # Получение данных
                data, addr = sock.recvfrom(4096)
                
                # Декодирование JSON
                json_data = json.loads(data.decode('utf-8'))
                
                frame_count += 1
                
                # Вывод данных
                print(f"\n=== Кадр {frame_count} ===")
                print(f"Timestamp: {json_data['timestamp']:.3f}")
                
                # Колени
                knee_pos = json_data['knee_positions']
                print(f"Левое колено: {knee_pos['left_knee']}, угол: {knee_pos['left_knee_angle']}")
                print(f"Правое колено: {knee_pos['right_knee']}, угол: {knee_pos['right_knee_angle']}")
                
                # Все joints (если есть)
                if 'joints' in json_data and json_data['joints']:
                    joints_count = len(json_data['joints'])
                    print(f"Всего joints: {joints_count}")
                
                # Путь штанги
                barbell_path = json_data['barbell_path']
                print(f"Точек пути штанги: {len(barbell_path)}")
                if barbell_path:
                    last_point = barbell_path[-1]
                    print(f"Последняя точка штанги: ({last_point['x']:.1f}, {last_point['y']:.1f})")
                
            except socket.timeout:
                # Таймаут - просто продолжаем
                continue
            except json.JSONDecodeError as e:
                print(f"Ошибка декодирования JSON: {e}")
            except Exception as e:
                print(f"Ошибка: {e}")
    
    except KeyboardInterrupt:
        print("\n\nОстановка получения данных")
    finally:
        sock.close()
        print("UDP сокет закрыт")

if __name__ == "__main__":
    main()

