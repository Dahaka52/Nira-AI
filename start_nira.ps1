param (
    [string]$config = "config",
    [string]$envPath = ""
)

# Очистка консоли
Clear-Host

Write-Host "=======================================" -ForegroundColor Cyan
Write-Host "    Nira UI: Система Запуска Проекта    " -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host ""

# Корневая директория проекта — абсолютный путь (не зависит от cwd)
$rootDir = (Get-Item -Path $PSScriptRoot).FullName

# Автовыбор окружения: .conda312 -> .conda
if ([string]::IsNullOrWhiteSpace($envPath)) {
    $preferred = Join-Path $rootDir ".conda312"
    $fallback = Join-Path $rootDir ".conda"
    if (Test-Path (Join-Path $preferred "python.exe")) {
        $envPath = $preferred
    } elseif (Test-Path (Join-Path $fallback "python.exe")) {
        $envPath = $fallback
    } else {
        throw "Python env not found. Expected '$preferred\\python.exe' or '$fallback\\python.exe'."
    }
} elseif (-not [System.IO.Path]::IsPathRooted($envPath)) {
    $envPath = Join-Path $rootDir $envPath
}

# Нормализуем путь (убираем .\ и приводим к каноническому абсолютному виду)
$envPath = (Resolve-Path -LiteralPath $envPath).Path

$pythonExe = Join-Path $envPath "python.exe"
if (-not (Test-Path $pythonExe)) {
    throw "python.exe not found in env path: $envPath"
}

# Настройка путей
$nodePath = "C:\Node.js"
if (Test-Path $nodePath) {
    $env:PATH = "$nodePath;" + $env:PATH
}

# Изоляция от user-site пакетов (устраняет конфликт версий из AppData)
$env:PYTHONNOUSERSITE = "1"

# Создаём папку logs если нет — Python упадёт если её нет
$logsDir = Join-Path $rootDir "logs"
if (-not (Test-Path $logsDir)) {
    New-Item -ItemType Directory -Path $logsDir | Out-Null
    Write-Host "[*] Создана папка logs" -ForegroundColor DarkGray
}

# 0. Очистка старых процессов (Kill Zombies)
Write-Host "[!] Очистка зависших процессов перед запуском..." -ForegroundColor Cyan
$processesToKill = @("llama-server", "node")
foreach ($p in $processesToKill) {
    Get-Process $p -ErrorAction SilentlyContinue | Stop-Process -Force
}

# Специальная очистка Python из выбранного окружения
Get-Process python -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -and $_.Path -like "*$envPath*" } |
    Stop-Process -Force

# Освобождаем ключевые порты проекта на случай "висячих" процессов без корректного Path
$portsToFree = @(3000, 6106, 7272)
Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
    Where-Object { $portsToFree -contains $_.LocalPort } |
    Select-Object -ExpandProperty OwningProcess -Unique |
    ForEach-Object {
        Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue
    }

Write-Host "[*] Очистка завершена." -ForegroundColor DarkGray
Write-Host ""

# 1. Запуск Backend
Write-Host "[*] Запуск backend из env: $envPath" -ForegroundColor Yellow
$backendAction = {
    param($config, $rootDir, $envPath)

    $env:PYTHONNOUSERSITE = "1"
    $env:CONDA_PREFIX = $envPath
    $env:CONDA_DEFAULT_ENV = $envPath
    $env:PATH = "$envPath;$envPath\Library\mingw-w64\bin;$envPath\Library\usr\bin;$envPath\Library\bin;$envPath\Scripts;$envPath\bin;" + $env:PATH

    & "$envPath\python.exe" "$rootDir\src\main.py" --config=$config --log_dir="$rootDir\logs"
}

# 2. Запуск Nira Web (Frontend)
Write-Host "[*] Подготовка и запуск Web Dashboard (Vite)..." -ForegroundColor Yellow
$frontendAction = {
    param($nodePath, $rootDir)
    if (Test-Path $nodePath) {
        $env:PATH = "$nodePath;" + $env:PATH
    }
    npm run dev -- --port 3000
}

# Запускаем backend первым — WorkingDirectory задаёт правильный cwd (PS7+)
$jobBackend = Start-Job -ScriptBlock $backendAction -ArgumentList $config, $rootDir, $envPath -WorkingDirectory $rootDir

Write-Host "[*] Ожидаем старт backend (5 сек)..." -ForegroundColor DarkGray
Start-Sleep -Seconds 5

# Запускаем frontend — WorkingDirectory для npm
$jobFrontend = Start-Job -ScriptBlock $frontendAction -ArgumentList $nodePath, $rootDir -WorkingDirectory "$rootDir\apps\nira-web"

Write-Host ""
Write-Host ">>> Система запущена!" -ForegroundColor Green
Write-Host ">>> Dashboard:  http://localhost:3000" -ForegroundColor Cyan
Write-Host ">>> API:        http://localhost:7272/api" -ForegroundColor Gray
Write-Host ">>> Логи:       $rootDir\logs\" -ForegroundColor Gray
Write-Host ">>> Env:        $envPath" -ForegroundColor Gray
Write-Host ">>> (Нажмите Ctrl+C для выхода)" -ForegroundColor Cyan
Write-Host "-------------------------------------------------------"

# Бесконечный цикл для вывода логов
try {
    while ($true) {
        $backLog = Receive-Job -Job $jobBackend
        if ($backLog) {
            $backLog | Write-Host
            $backLog | Out-File -Append -FilePath "$rootDir\logs\backend_console_$(Get-Date -Format 'yyyy-MM-dd').log" -Encoding UTF8
        }

        $frontLog = Receive-Job -Job $jobFrontend
        if ($frontLog) {
            $frontLog | Write-Host
            $frontLog | Out-File -Append -FilePath "$rootDir\logs\frontend_vite_$(Get-Date -Format 'yyyy-MM-dd').log" -Encoding UTF8
        }
        Start-Sleep -Milliseconds 500
    }
}
finally {
    # Очистка при выходе
    Write-Host "`n[*] Остановка системных процессов..." -ForegroundColor Red
    Stop-Job $jobBackend -ErrorAction SilentlyContinue
    Stop-Job $jobFrontend -ErrorAction SilentlyContinue
    Remove-Job $jobBackend -ErrorAction SilentlyContinue
    Remove-Job $jobFrontend -ErrorAction SilentlyContinue
}
