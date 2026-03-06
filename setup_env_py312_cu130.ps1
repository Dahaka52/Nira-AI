param (
    [string]$EnvPath = ".\\.conda312",
    [string]$FlashAttnWheelUrl = "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.12/flash_attn-2.8.3%2Bcu130torch2.10-cp312-cp312-win_amd64.whl",
    [switch]$SkipFlashAttn,
    [switch]$SkipSpacyModel,
    [switch]$SkipUnidic,
    [switch]$SkipPipCheck
)

$ErrorActionPreference = "Stop"

$rootDir = (Get-Item -Path $PSScriptRoot).FullName
if ([System.IO.Path]::IsPathRooted($EnvPath)) {
    $envPrefix = $EnvPath
} else {
    $envPrefix = Join-Path $rootDir $EnvPath
}
$pythonExe = Join-Path $envPrefix "python.exe"

$condaCmd = Get-Command conda -ErrorAction SilentlyContinue
if ($condaCmd) {
    $condaBin = $condaCmd.Source
} elseif (Test-Path "C:\\Miniconda3\\Scripts\\conda.exe") {
    $condaBin = "C:\\Miniconda3\\Scripts\\conda.exe"
} else {
    throw "Conda not found in PATH and C:\\Miniconda3\\Scripts\\conda.exe not found."
}

Write-Host "[1/7] Target env prefix: $envPrefix" -ForegroundColor Cyan
if (-not (Test-Path $pythonExe)) {
    Write-Host "Creating conda env with Python 3.12..." -ForegroundColor Yellow
    & $condaBin create -y -p $envPrefix python=3.12 pip
} else {
    Write-Host "Conda env already exists, reusing it." -ForegroundColor DarkGray
}

Write-Host "[2/7] Verifying Python..." -ForegroundColor Cyan
& $pythonExe --version

# Keep runtime isolated from user-site packages.
$env:PYTHONNOUSERSITE = "1"

Write-Host "[3/7] Upgrading pip toolchain..." -ForegroundColor Cyan
& $pythonExe -m pip install --upgrade pip setuptools wheel

Write-Host "[4/7] Installing PyTorch cu130..." -ForegroundColor Cyan
& $pythonExe -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130

if (-not $SkipFlashAttn) {
    Write-Host "[5/7] Installing flash-attn wheel and triton-windows..." -ForegroundColor Cyan
    & $pythonExe -m pip install $FlashAttnWheelUrl
    & $pythonExe -m pip install -U "triton-windows<3.7"
} else {
    Write-Host "[5/7] Skipping flash-attn installation by request." -ForegroundColor DarkGray
}

Write-Host "[6/7] Installing project dependencies..." -ForegroundColor Cyan
& $pythonExe -m pip install -r (Join-Path $rootDir "requirements.txt")
& $pythonExe -m pip install --no-deps -r (Join-Path $rootDir "requirements.no_deps.txt")

if (-not $SkipSpacyModel) {
    Write-Host "[6.1] Downloading spacy model..." -ForegroundColor Cyan
    & $pythonExe -m spacy download en_core_web_sm
}
if (-not $SkipUnidic) {
    Write-Host "[6.2] Downloading unidic..." -ForegroundColor Cyan
    & $pythonExe -m unidic download
}

Write-Host "[7/7] Validation..." -ForegroundColor Cyan
& $pythonExe -c "import torch;print('torch', torch.__version__);print('cuda', torch.version.cuda);print('is_available', torch.cuda.is_available())"
if (-not $SkipPipCheck) {
    & $pythonExe -m pip check
}

Write-Host ""
Write-Host "Environment is ready: $envPrefix" -ForegroundColor Green
Write-Host "Use start_nira.ps1 -envPath `"$envPrefix`" to run backend from this env." -ForegroundColor Green
