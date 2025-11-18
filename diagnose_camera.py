"""Диагностика камеры Blackmagic через OpenCV"""
import cv2
import numpy as np
import sys

def test_camera(idx):
    """Тестирование камеры с разными настройками"""
    print(f"\n{'='*60}")
    print(f"Тестирование камеры {idx}")
    print(f"{'='*60}")
    
    # Вариант 1: DirectShow с настройками
    print("\n1. DirectShow с настройками разрешения...")
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if cap.isOpened():
        # Пробуем установить параметры
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 50)
        
        # Читаем несколько кадров
        for i in range(10):
            ret, frame = cap.read()
            if ret and frame is not None:
                mean = frame.mean()
                print(f"  Кадр {i+1}: размер={frame.shape}, mean={mean:.2f}")
                if mean > 1.0:
                    print(f"  ✅ НАЙДЕН СИГНАЛ! Средняя яркость: {mean:.2f}")
                    cap.release()
                    return True
        cap.release()
        print("  ❌ Все кадры черные")
    else:
        print("  ❌ Не удалось открыть")
    
    # Вариант 2: Media Foundation
    print("\n2. Media Foundation...")
    try:
        cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            for i in range(10):
                ret, frame = cap.read()
                if ret and frame is not None:
                    mean = frame.mean()
                    print(f"  Кадр {i+1}: размер={frame.shape}, mean={mean:.2f}")
                    if mean > 1.0:
                        print(f"  ✅ НАЙДЕН СИГНАЛ! Средняя яркость: {mean:.2f}")
                        cap.release()
                        return True
            cap.release()
            print("  ❌ Все кадры черные")
        else:
            print("  ❌ Не удалось открыть")
    except Exception as e:
        print(f"  ❌ Ошибка: {e}")
    
    # Вариант 3: DirectShow без настроек (авто)
    print("\n3. DirectShow без настроек (автоматический формат)...")
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if cap.isOpened():
        for i in range(10):
            ret, frame = cap.read()
            if ret and frame is not None:
                mean = frame.mean()
                print(f"  Кадр {i+1}: размер={frame.shape}, mean={mean:.2f}")
                if mean > 1.0:
                    print(f"  ✅ НАЙДЕН СИГНАЛ! Средняя яркость: {mean:.2f}")
                    cap.release()
                    return True
        cap.release()
        print("  ❌ Все кадры черные")
    
    return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        idx = int(sys.argv[1])
    else:
        idx = 0
    
    print("🔍 ДИАГНОСТИКА КАМЕРЫ BLACKMAGIC DECKLINK")
    print("="*60)
    print("\n⚠️ ВАЖНО:")
    print("1. Закройте OBS и другие приложения, использующие камеру")
    print("2. Проверьте настройки Blackmagic Desktop Video:")
    print("   - Откройте Blackmagic Desktop Video Setup")
    print("   - Убедитесь, что 'WDM Capture' включен")
    print("   - Выберите правильный формат входного сигнала")
    print("3. Убедитесь, что на SDI входе есть активный сигнал")
    print("="*60)
    
    success = test_camera(idx)
    
    print(f"\n{'='*60}")
    if success:
        print("✅ Камера работает! Сигнал обнаружен.")
    else:
        print("❌ Камера не дает видеосигнал через OpenCV")
        print("\n💡 РЕКОМЕНДАЦИИ:")
        print("1. Проверьте Blackmagic Desktop Video Setup:")
        print("   - Включите 'WDM Capture'")
        print("   - Выберите правильный формат (1080p50, 1080i50, и т.д.)")
        print("2. Перезапустите Blackmagic Desktop Video Service")
        print("3. Попробуйте использовать OBS Virtual Camera как обходной путь")
        print("4. Или используйте ffmpeg с поддержкой DeckLink")
    print(f"{'='*60}\n")

