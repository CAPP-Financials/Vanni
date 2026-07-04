# Vanni launcher / lifecycle manager.
#   .\Vanni-launcher.ps1            -> start Ollama (if down) + Vanni.exe, both hidden
#   .\Vanni-launcher.ps1 -Elevated  -> start Vanni as admin (so it can paste into
#                                       elevated windows); triggers a UAC prompt
#   .\Vanni-launcher.ps1 -Stop      -> cleanly terminate Vanni (and Ollama if we started it)
# Double-click helpers: make shortcuts running
#   powershell -ExecutionPolicy Bypass -File Vanni-launcher.ps1 [-Stop] [-Elevated]
param([switch]$Stop, [switch]$Elevated)

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

if ($Elevated) {
    # RunAs triggers a UAC prompt; an elevated Vanni can paste into admin windows.
    # Can't combine with -WindowStyle Hidden (UAC needs interaction).
    Start-Process $vanniExe -Verb RunAs -WorkingDirectory (Split-Path $vanniExe)
    Write-Host "Vanni started elevated (admin). Hold Ctrl+Win to dictate. Stop with: .\Vanni-launcher.ps1 -Stop"
} else {
    Start-Process $vanniExe -WindowStyle Hidden -WorkingDirectory (Split-Path $vanniExe)
    Write-Host "Vanni started (hidden). Hold Ctrl+Win to dictate. Stop with: .\Vanni-launcher.ps1 -Stop"
}
