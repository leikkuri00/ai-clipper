param(
    [string]$ModelName = "qwen2.5-coder",
    [string]$Url = "",
    [string]$Dest = "models",
    [switch]$Confirm
)

Set-StrictMode -Version Latest

function Prompt-LicensingWarning {
    Write-Host "WARNING: Model weights are large and often have license restrictions." -ForegroundColor Yellow
    Write-Host "You must have the right to download and use the model. The script will not download a model unless you explicitly confirm." -ForegroundColor Yellow
}

if (-not (Test-Path $Dest)) { New-Item -ItemType Directory -Path $Dest | Out-Null }

Prompt-LicensingWarning

if ([string]::IsNullOrWhiteSpace($Url)) {
    Write-Host "No URL provided. To download a model, run this script with -Url '<direct-download-url>' and -Confirm to proceed." -ForegroundColor Cyan
    Write-Host "Examples:" -ForegroundColor Gray
    Write-Host "  .\download-model.ps1 -ModelName qwen2.5-coder -Url 'https://example.com/qwen2.5.gguf' -Confirm"
    Write-Host "If you don't have a direct URL, obtain it from the model provider's official release page and re-run with -Url." -ForegroundColor Gray
    exit 1
}

$outDir = Join-Path -Path $Dest -ChildPath $ModelName
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }

$fileName = Split-Path -Path $Url -Leaf
$targetPath = Join-Path -Path $outDir -ChildPath $fileName

if (-not $Confirm) {
    Write-Host "Download not started. Re-run with -Confirm to actually download the model to:`n  $targetPath" -ForegroundColor Yellow
    exit 1
}

Write-Host "Starting download of $ModelName to $targetPath"

try {
    # Use BITS for resumable transfer on Windows
    if (Get-Command Start-BitsTransfer -ErrorAction SilentlyContinue) {
        Start-BitsTransfer -Source $Url -Destination $targetPath -Description "Downloading $ModelName" -DisplayName "download-model"
    } else {
        # Fallback to Invoke-WebRequest
        Invoke-WebRequest -Uri $Url -OutFile $targetPath -UseBasicParsing
    }
    Write-Host "Download completed: $targetPath" -ForegroundColor Green
} catch {
    Write-Host "Download failed: $_" -ForegroundColor Red
    exit 2
}

# Basic validation: file exists and size > 0
if ((Test-Path $targetPath) -and ((Get-Item $targetPath).Length -gt 0)) {
    Write-Host "Model saved to $targetPath" -ForegroundColor Green
} else {
    Write-Host "Model file missing or empty after download." -ForegroundColor Red
    exit 3
}
