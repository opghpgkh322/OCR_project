@echo off
setlocal

REM Сборка desktop-приложения в .exe
REM Запускать в окружении, где установлены зависимости проекта + pyinstaller

pyinstaller --noconfirm --windowed --name OCRDesk scripts\desktop_app.py

echo.
echo Build finished. See dist\OCRDesk\OCRDesk.exe
pause