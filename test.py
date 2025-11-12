"""Тест обнаружения камер с реальным видеосигналом"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from unified_main import list_cameras

print("=" * 60)
print("Поиск камер с реальным видеосигналом...")
print("=" * 60)

cams = list_cameras(max_test=6)

print("=" * 60)
print(f"Найдено активных камер: {len(cams)}")
if cams:
    print(f"Индексы: {cams}")
else:
    print("⚠️ Активных камер не найдено!")
print("=" * 60)