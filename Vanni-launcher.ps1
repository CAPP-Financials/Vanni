# Vanni launcher / lifecycle manager.
#   .\Vanni-launcher.ps1          -> start Ollama (if down) + Vanni.exe, both hidden
#   .\Vanni-launcher.ps1 -Stop    -> cleanly terminate Vanni (and Ollama if we started it)
# Double-click helpers: make shortcuts running
#   powershell -ExecutionPolicy Bypass -File Vanni-launcher.ps1 [-Stop]
param([switch]$Stop)

$ErrorActionPreference = "SilentlyContinue"
$vanniExe  = Join-Path $PSScriptRoot "dist\Vanni\Vanni.exe"
$stateFile = Join-Path $env:TEMP "vanni-launcher-state.json"

function Test-Ollama { (Test-NetConnection 127.0.0.1 -Port 11434 -InformationLevel Quiet -WarningAction SilentlyContinue) }

if ($Stop) {
    $state = if (Test-Path $stateFile) { Get-Content $stateFile | ConvertFrom-Json } else { $null }
    Get-Process Vanni -ErrorAction SilentlyContinue | Stop-Process -Force
    if ($state -and $state.startedOllama) {
        Get-Process ollama* -ErrorAction SilentlyContinue | Stop-Process -Force
        Write-Host "Vanni and Ollama stopped."
    } else {
        Write-Host "Vanni stopped (Ollama left running - it was already up)."
    }
    Remove-Item $stateFile -ErrorAction SilentlyContinue
    exit 0
}

if (-not (Test-Path $vanniExe)) { Write-Error "Vanni.exe not found at $vanniExe - run the PyInstaller build first."; exit 1 }
if (Get-Process Vanni -ErrorAction SilentlyContinue) { Write-Host "Vanni is already running."; exit 0 }

$startedOllama = $false
if (-not (Test-Ollama)) {
    Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
    $startedOllama = $true
    foreach ($i in 1..40) { if (Test-Ollama) { break }; Start-Sleep -Milliseconds 500 }
}
@{ startedOllama = $startedOllama } | ConvertTo-Json | Set-Content $stateFile

Start-Process $vanniExe -WindowStyle Hidden -WorkingDirectory (Split-Path $vanniExe)
Write-Host "Vanni started (hidden). Hold Ctrl+Win to dictate. Stop with: .\Vanni-launcher.ps1 -Stop"
