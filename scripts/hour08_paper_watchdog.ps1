# Keeps the hour-08 paper trader alive. Paper only -- it places no real orders.
# The strategy fires once per day inside a 24-minute window (04:06-04:30 UTC), so an
# outage of even an hour silently costs a full observation from a sample that can
# only grow at one per day. This runs every 5 minutes and restarts if it is gone.
$ErrorActionPreference = "Continue"
$repo = "D:\Finance\polymarket"
$log  = "$repo\results\hour08_paper\wrapper_log.txt"
$stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like '*run_hour08_paper*' }
if ($running) { exit 0 }
New-Item -ItemType Directory -Force -Path "$repo\results\hour08_paper" | Out-Null
Add-Content $log "$stamp watchdog: not running, starting a new instance"
Start-Process -FilePath "python" -ArgumentList "$repo\scripts\run_hour08_paper.py" `
    -WorkingDirectory $repo -WindowStyle Hidden
