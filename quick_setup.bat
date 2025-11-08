@echo off
REM Быстрая настройка проекта после установки Python 3.10
echo ========================================
echo Быстрая настройка проекта DartHeavyLifting
echo ========================================
echo.

REM Проверка Python 3.10
echo [1/4] Проверка Python 3.10...
py -3.10 --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python 3.10 не найден!
    echo.
    echo Установите Python 3.10 одним из способов:
    echo 1. Запустите setup_python310.bat
    echo 2. Установите вручную с python.org
    echo 3. Используйте: winget install Python.Python.3.10
    echo.
    pause
    exit /b 1
)

py -3.10 --version
echo.

REM Создание виртуального окружения
echo [2/4] Создание виртуального окружения...
if exist venv (
    echo Виртуальное окружение уже существует. Пропускаем создание.
) else (
    py -3.10 -m venv venv
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Не удалось создать виртуальное окружение.
        pause
        exit /b 1
    )
    echo Виртуальное окружение создано.
)
echo.

REM Активация и установка зависимостей
echo [3/4] Активация окружения и обновление pip...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
echo.

echo [4/4] Установка зависимостей проекта...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Ошибка при установке зависимостей.
    echo Проверьте файл requirements.txt и попробуйте снова.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Настройка завершена успешно!
echo ========================================
echo.
echo Для запуска проекта:
echo   1. Активируйте окружение: venv\Scripts\activate
echo   2. Запустите: python main.py
echo.
echo Или просто запустите: python main.py
echo (если окружение уже активировано)
echo.
pause

