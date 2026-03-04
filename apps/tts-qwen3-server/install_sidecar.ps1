param(
    [string]$BasePython = "C:\Nirmita\.conda\python.exe",
    [string]$VenvName = ".venv",
    [switch]$InstallFlashAttn
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $Root $VenvName
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$ReqFile = Join-Path $Root "requirements.sidecar.txt"

if (-not (Test-Path $BasePython)) {
    throw "Base python not found: $BasePython"
}

if (-not (Test-Path $VenvPython)) {
    & $BasePython -m venv $VenvDir
}

& $VenvPython -m pip install --upgrade pip setuptools wheel

if ($VenvName -eq ".venv312") {
    & $VenvPython -m pip install torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130
}

& $VenvPython -m pip install -r $ReqFile

if ($InstallFlashAttn.IsPresent -or $VenvName -eq ".venv312") {
    & $VenvPython -m pip install `
        "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.12/flash_attn-2.8.3%2Bcu130torch2.10-cp312-cp312-win_amd64.whl"
}

Write-Host "Qwen3 sidecar env ready: $VenvPython"
Write-Host "Tip: set process.python_executable in config to this path."
