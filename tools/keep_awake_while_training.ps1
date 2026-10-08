# Keeps Windows from idle-sleeping while the training process runs (like a video player does).
# Changes no settings; the request ends automatically when this script exits.
Add-Type -Namespace W -Name P -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
$ES_CONTINUOUS = [uint32]"0x80000000"; $ES_SYSTEM_REQUIRED = [uint32]1
[W.P]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED) | Out-Null
"keep-awake on $(Get-Date)"
while (Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*train_cpu*' }) {
    Start-Sleep -Seconds 60
}
[W.P]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
"training finished, keep-awake off $(Get-Date)"
