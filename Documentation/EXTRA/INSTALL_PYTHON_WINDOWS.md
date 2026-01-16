# Установка Python 3.10/3.11 на Windows через Python Installation Manager

## Использование Python Launcher (py)

Python Launcher (`py`) - это встроенный инструмент Windows для управления несколькими версиями Python.

### Шаг 1: Проверка наличия Python Launcher

Откройте командную строку или PowerShell и проверьте:
```bash
py --version
```

Если команда не найдена, установите Python с официального сайта (при установке Python launcher устанавливается автоматически).

### Шаг 2: Установка Python 3.10

#### Способ 1: Скачать и установить вручную
1. Перейдите на https://www.python.org/downloads/release/python-31012/
2. Скачайте установщик для Windows (Windows installer (64-bit))
3. Запустите установщик
4. **ВАЖНО:** При установке отметьте "Add Python 3.10 to PATH"
5. При установке выберите "Install for all users" (опционально)

#### Способ 2: Использование winget (Windows Package Manager)
Если у вас установлен Windows Package Manager:
```bash
winget install Python.Python.3.10
```

### Шаг 3: Проверка установки

После установки проверьте доступные версии Python:
```bash
py --list
```

Вы должны увидеть что-то вроде:
```
-V:3.13 *        Python 3.13.2
-V:3.10          Python 3.10.12
```

### Шаг 4: Создание виртуального окружения с Python 3.10

Используйте Python launcher для создания виртуального окружения с нужной версией:

```bash
# Создать виртуальное окружение с Python 3.10
py -3.10 -m venv venv

# Активировать виртуальное окружение
venv\Scripts\activate

# Проверить версию Python в окружении
python --version
# Должно показать: Python 3.10.x

# Обновить pip
python -m pip install --upgrade pip

# Установить зависимости
pip install -r requirements.txt
```

### Шаг 5: Использование проекта

Теперь вы можете работать с проектом:
```bash
# Активировать окружение (если еще не активировано)
venv\Scripts\activate

# Запустить проект
python main.py
```

## Альтернативный способ: Указание версии при каждом запуске

Если не хотите создавать виртуальное окружение, можете использовать Python launcher напрямую:

```bash
# Установка зависимостей с Python 3.10
py -3.10 -m pip install -r requirements.txt

# Запуск скрипта с Python 3.10
py -3.10 main.py
```

## Решение проблем

### Python 3.10 не появляется в списке после установки

1. Убедитесь, что Python 3.10 установлен правильно
2. Перезапустите командную строку/PowerShell
3. Проверьте путь установки: `py -3.10 --version`

### Ошибка "Python was not found"

Убедитесь, что Python 3.10 добавлен в PATH:
1. Откройте "Системные переменные среды"
2. Проверьте переменную PATH
3. Должен быть путь к Python 3.10 (например, `C:\Python310\` или `C:\Users\Username\AppData\Local\Programs\Python\Python310\`)

### Несколько версий Python

Python launcher позволяет использовать разные версии:
- `py -3.13` - использовать Python 3.13
- `py -3.10` - использовать Python 3.10
- `py -3` - использовать последнюю версию Python 3.x
- `py` - использовать версию по умолчанию

## Автоматическая установка (скрипты)

Я создал два скрипта для автоматизации:

### 1. setup_python310.bat
Устанавливает Python 3.10 через winget и создает виртуальное окружение:
```bash
setup_python310.bat
```

### 2. quick_setup.bat
Быстрая настройка проекта (создает окружение и устанавливает зависимости):
```bash
quick_setup.bat
```

## Быстрая команда для начала работы

```bash
# Создать окружение и установить зависимости одной командой
py -3.10 -m venv venv && venv\Scripts\activate && pip install -r requirements.txt
```

