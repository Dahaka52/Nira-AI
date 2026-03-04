param(
    [string]$BasePython = "C:\Nirmita\.conda\python.exe"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$ReqFile = Join-Path $Root "requirements.sidecar.txt"

if (-not (Test-Path $BasePython)) {
    throw "Base python not found: $BasePython"
}

if (-not (Test-Path $VenvPython)) {
    & $BasePython -m venv --system-site-packages (Join-Path $Root ".venv")
}

& $VenvPython -m pip install --upgrade pip setuptools wheel
& $VenvPython -m pip install -r $ReqFile

Write-Host "Qwen3 sidecar env ready: $VenvPython"
Write-Host "Tip: set process.python_executable in config to this path."

