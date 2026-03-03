@echo off
chcp 65001 > nul
cd /d "%~dp0"

echo [*] Запуск Nira UI (Обход ExecutionPolicy)...
pwsh -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_nira.ps1"
pause
