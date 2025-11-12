"""Тест обнаружения камер через OpenCV"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from unified_main import list_cameras

print("=" * 60)
print("Поиск доступных камер через OpenCV...")
print("=" * 60)

cams = list_cameras(max_test=6)

print("=" * 60)
print(f"Найдено камер: {len(cams)}")
if cams:
    print(f"Индексы: {cams}")
else:
    print("⚠️ Камер не найдено!")
print("=" * 60)