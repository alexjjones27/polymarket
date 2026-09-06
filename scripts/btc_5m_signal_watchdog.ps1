# Watchdog for run_btc_5m_signal.py, invoked every 5 minutes by the
# PolymarketBTC5mSignalWatchdog scheduled task (a plain time-based
# trigger, since ONLOGON/ONSTART triggers are denied in this environment).
# run_btc_5m_signal.py itself loops forever -- this just checks whether a
# copy is already running and, if not, starts a new detached one. State
# (results/btc_5m_live/state.json) persists across restarts, so this is
# safe to run redundantly.

$repoRoot = "D:\Finance\polymarket"
$python = "C:\Users\AlexJ\AppData\Local\Programs\Python\Python312\python.exe"
$scriptPath = "$repoRoot\scripts\run_btc_5m_signal.py"
$logDir = "$repoRoot\results\btc_5m_live"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$wrapperLog = "$logDir\wrapper_log.txt"

$existing = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -like "*run_btc_5m_signal.py*" }

if ($existing) {
    exit 0
}

Add-Content -Path $wrapperLog -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') watchdog: not running, starting a new instance"
Start-Process -FilePath $python -ArgumentList "`"$scriptPath`"" -WorkingDirectory $repoRoot -WindowStyle Hidden
