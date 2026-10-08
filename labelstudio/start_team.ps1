# Start Label Studio for the whole team on the lab network (LAN only - never forward this port to the Internet).
#   .\labelstudio\start_team.ps1
# Colleagues open http://<this-PC's-LAN-address>:8080 . Sign-up is limited to invite links: in Label Studio
# open Organization > "Add People", copy the invite link and send it to each labeller.
# Windows asks once to allow Python through the firewall: allow it for PRIVATE networks only.
$root = Split-Path -Parent $PSScriptRoot
$ip = (Get-NetIPAddress -AddressFamily IPv4 |
       Where-Object { $_.PrefixOrigin -in 'Dhcp', 'Manual' -and $_.IPAddress -notlike '169.254*' } |
       Select-Object -First 1).IPAddress
if (-not $ip) { $ip = "localhost" }
$env:LABEL_STUDIO_BASE_DATA_DIR = Join-Path $root "labelstudio\data"
$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED = "true"
$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT = $root
$env:LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK = "true"      # only people with an invite link can join
$env:LABEL_STUDIO_HOST = "http://${ip}:8080"
Write-Host "Team Label Studio: http://${ip}:8080   (invite people via Organization > Add People)"
& (Join-Path $root ".labelstudio-venv\Scripts\python.exe") -m label_studio.server start --port 8080 --internal-host 0.0.0.0
