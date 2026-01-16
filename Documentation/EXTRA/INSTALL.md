# Инструкция по установке

## Установка зависимостей

### Стандартная установка
```bash
pip install -r requirements.txt
```

### Если возникают проблемы с MediaPipe

MediaPipe может требовать определенную версию Python и может иметь проблемы на некоторых системах.

#### Вариант 1: Установка MediaPipe отдельно
```bash
pip install opencv-python numpy
pip install mediapipe
```

#### Вариант 2: Установка конкретной версии MediaPipe
```bash
pip install opencv-python numpy
pip install mediapipe==0.10.9
```

#### Вариант 3: Установка через conda (рекомендуется для Windows)
```bash
conda install -c conda-forge opencv numpy
pip install mediapipe
```

## Требования к системе

- **Python 3.8, 3.9, 3.10 или 3.11** (рекомендуется 3.10)
  - ⚠️ **ВНИМАНИЕ:** Python 3.12+ НЕ поддерживается MediaPipe!
  - Если у вас установлена более новая версия Python, используйте одну из поддерживаемых версий
- Windows 10/11, Linux или macOS
- Для MediaPipe требуется 64-битная система

### Установка правильной версии Python

#### Вариант 1: Использование pyenv (рекомендуется)
```bash
# Установите pyenv для Windows (pyenv-win)
# Затем установите Python 3.10:
pyenv install 3.10.12
pyenv local 3.10.12
```

#### Вариант 2: Использование conda
```bash
conda create -n dartlifting python=3.10
conda activate dartlifting
pip install -r requirements.txt
```

#### Вариант 3: Установка Python 3.10 отдельно
1. Скачайте Python 3.10 с официального сайта
2. Установите его в отдельную директорию
3. Используйте полный путь к Python 3.10 для создания виртуального окружения

## Проверка установки

После установки проверьте:
```bash
python -c "import cv2; import mediapipe; import numpy; print('Все модули установлены успешно')"
```

Если MediaPipe не устанавливается, попробуйте:
1. Обновить pip: `python -m pip install --upgrade pip`
2. Установить Microsoft Visual C++ Redistributable (для Windows)
3. Использовать виртуальное окружение Python

