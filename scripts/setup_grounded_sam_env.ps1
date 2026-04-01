param(
    [string]$EnvName = "grounded_sam_rgbd",
    [string]$PythonVersion = "3.10",
    [string]$RepoRoot = "third_party/Grounded-Segment-Anything",
    [switch]$InstallRepoBackend
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "Creating conda environment: $EnvName (python=$PythonVersion)"
conda create -y -n $EnvName python=$PythonVersion

Write-Host "Installing PyTorch with CUDA 11.8"
conda install -y -n $EnvName pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

Write-Host "Installing Python packages"
conda run -n $EnvName python -m pip install --upgrade pip
conda run -n $EnvName python -m pip install -r requirements-grounded-sam.txt

Write-Host "Installing official Segment Anything package from cloned repo"
conda run -n $EnvName python -m pip install -e "$RepoRoot\segment_anything"

if ($InstallRepoBackend) {
    Write-Host "Trying to install official GroundingDINO repo backend"
    try {
        conda run -n $EnvName python -m pip install wheel
        conda run -n $EnvName python -m pip install --no-build-isolation -e "$RepoRoot\GroundingDINO"
        Write-Host "GroundingDINO repo backend installed successfully"
    }
    catch {
        Write-Warning "GroundingDINO repo backend install failed. The project can still run with --backend transformers or --backend auto."
    }
}
else {
    Write-Host "Skipping GroundingDINO repo backend install. Use -InstallRepoBackend if you want the compiled official backend."
}

Write-Host "Exporting environment summary"
conda run -n $EnvName python scripts\export_runtime_env.py --output outputs\runtime_environment.txt

Write-Host "Environment setup finished."
Write-Host "Activate with: conda activate $EnvName"
