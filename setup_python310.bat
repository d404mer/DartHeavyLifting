@echo off
REM Скрипт для установки Python 3.10 и настройки окружения для проекта
echo ========================================
echo Установка Python 3.10 для DartHeavyLifting
echo ========================================
echo.

REM Проверка наличия winget
where winget >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] winget не найден. Установка через winget невозможна.
    echo Пожалуйста, установите Python 3.10 вручную с python.org
    echo Или используйте инструкцию из INSTALL_PYTHON_WINDOWS.md
    pause
    exit /b 1
)

echo [1/5] Проверка установленных версий Python...
py --list
echo.

echo [2/5] Установка Python 3.10 через winget...
winget install Python.Python.3.10 --silent --accept-package-agreements --accept-source-agreements
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Ошибка при установке через winget. Попробуйте установить вручную.
    echo Скачайте Python 3.10 с: https://www.python.org/downloads/release/python-31012/
    pause
    exit /b 1
)

echo.
echo [3/5] Ожидание завершения установки...
timeout /t 5 /nobreak >nul

echo.
echo [4/5] Проверка установки Python 3.10...
py -3.10 --version
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python 3.10 не найден после установки.
    echo Попробуйте перезапустить командную строку и запустить скрипт снова.
    pause
    exit /b 1
)

echo.
echo [5/5] Создание виртуального окружения...
py -3.10 -m venv venv
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Не удалось создать виртуальное окружение.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Установка завершена успешно!
echo ========================================
echo.
echo Следующие шаги:
echo 1. Активируйте виртуальное окружение: venv\Scripts\activate
echo 2. Установите зависимости: pip install -r requirements.txt
echo 3. Запустите проект: python main.py
echo.
echo Или используйте команду:
echo   venv\Scripts\activate && pip install -r requirements.txt
echo.
pause

