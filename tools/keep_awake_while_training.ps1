# Keeps Windows from idle-sleeping while the training process runs (like a video player does).
# Changes no settings; the request ends automatically when this script exits.
Add-Type -Namespace W -Name P -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
$ES_CONTINUOUS = [uint32]"0x80000000"; $ES_SYSTEM_REQUIRED = [uint32]1
# Modern Standby laptops still sleep when the screen turns off: also ask for the display (it may dim)
$ES_DISPLAY_REQUIRED = [uint32]2
[W.P]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED -bor $ES_DISPLAY_REQUIRED) | Out-Null
"keep-awake on $(Get-Date)"
while (Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'train_cpu|gen3.py|eval_app|export_formats|train_tek_classifier' }) {
    Start-Sleep -Seconds 60
}
[W.P]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
"training finished, keep-awake off $(Get-Date)"
