"""
Скрипт для проверки совместимости версии Python с MediaPipe
"""

import sys

def check_python_version():
    """Проверка версии Python"""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    print(f"Текущая версия Python: {version_str}")
    
    # MediaPipe поддерживает Python 3.8-3.11
    if version.major == 3 and 8 <= version.minor <= 11:
        print("[OK] Версия Python совместима с MediaPipe!")
        return True
    else:
        print("[ERROR] Версия Python НЕ совместима с MediaPipe!")
        print("\nMediaPipe поддерживает только Python 3.8-3.11")
        print("\nРешения:")
        print("1. Установите Python 3.10 или 3.11")
        print("2. Используйте виртуальное окружение с нужной версией Python")
        print("3. Используйте conda: conda create -n dartlifting python=3.10")
        return False

if __name__ == "__main__":
    check_python_version()

