# Start the recognition server for the Quest / Unity client (Option A, local network only).
#   .\tools\start_lan_server.ps1            # normal
#   .\tools\start_lan_server.ps1 -Capture   # also keep unsure frames from clients that opted in (for labelling)
# Never forward port 8011 to the Internet.
param([switch]$Capture, [int]$Port = 8011)
$root = Split-Path -Parent $PSScriptRoot
Write-Host "Server addresses for Unity (Server Url):"
Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254*' } |
  ForEach-Object { $tag = if ($_.IPAddress -eq '192.168.137.1') { '  <- Windows Mobile Hotspot (recommended for the Quest)' } else { '' }
                   Write-Host ("  http://{0}:{1}   ({2}){3}" -f $_.IPAddress, $Port, $_.InterfaceAlias, $tag) }
Write-Host "Test from the headset browser: http://<address>:$Port/health"
if ($Capture) { $env:CAPTURE_MODE = "on" }
& (Join-Path $root ".venv\Scripts\python.exe") -m uvicorn server.app:app --app-dir $root --host 0.0.0.0 --port $Port
