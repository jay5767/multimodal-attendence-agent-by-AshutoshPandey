# Cloudflare Tunnel Launcher for Instant Global HTTPS Access
# No account or signup needed. Generates a secure https://*.trycloudflare.com URL.

param (
    [int]$Port = 5000
)

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  Multimodal Attendance & Engagement Agent - Public Web Access" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Check if the local app is listening on the port
$tcpConnection = Test-NetConnection -ComputerName 127.0.0.1 -Port $Port -InformationLevel Quiet -WarningAction SilentlyContinue

if (-not $tcpConnection) {
    Write-Host "[INFO] App is not running on port $Port yet. Starting app in background..." -ForegroundColor Yellow
    $py311 = "C:\Users\Ashutoshpandey\AppData\Local\Programs\Python\Python311\python.exe"
    $pyCmd = if (Test-Path $py311) { $py311 } else { "python" }
    
    Start-Process -FilePath $pyCmd -ArgumentList "app.py" -WindowStyle Minimized
    Write-Host "[INFO] Waiting 5 seconds for server to initialize..." -ForegroundColor Yellow
    Start-Sleep -Seconds 5
}

# 2. Check or download cloudflared
$cloudflaredPath = Join-Path $PSScriptRoot "cloudflared.exe"

if (-not (Test-Path $cloudflaredPath)) {
    Write-Host "[INFO] Downloading Cloudflare Tunnel (cloudflared.exe)..." -ForegroundColor Cyan
    $downloadUrl = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
    try {
        Invoke-WebRequest -Uri $downloadUrl -OutFile $cloudflaredPath -UseBasicParsing
        Write-Host "[SUCCESS] Cloudflared downloaded successfully." -ForegroundColor Green
    } catch {
        Write-Host "[WARNING] Cloudflared direct download failed. Trying npx localtunnel..." -ForegroundColor Yellow
        if (Get-Command npx -ErrorAction SilentlyContinue) {
            Write-Host "[INFO] Running: npx localtunnel --port $Port" -ForegroundColor Green
            npx localtunnel --port $Port
            exit
        } else {
            Write-Host "[ERROR] Could not download cloudflared or run npx." -ForegroundColor Red
            exit 1
        }
    }
}

Write-Host ""
Write-Host "----------------------------------------------------------------------" -ForegroundColor Green
Write-Host "  CREATING SECURE HTTPS TUNNEL... PLEASE WAIT A MOMENT." -ForegroundColor Green
Write-Host "  Your live website link will appear below (https://*.trycloudflare.com)" -ForegroundColor Green
Write-Host "  Anyone anywhere in the world can open this link on their phone/PC!" -ForegroundColor Green
Write-Host "----------------------------------------------------------------------" -ForegroundColor Green
Write-Host ""

& $cloudflaredPath tunnel --url "http://localhost:$Port"

