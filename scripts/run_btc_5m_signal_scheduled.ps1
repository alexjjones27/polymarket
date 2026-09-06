# Wrapper for Windows Task Scheduler (task name: PolymarketBTC5mSignal).
# run_btc_5m_signal.py itself loops forever, checking every 5-minute BTC
# window at T-30s -- this wrapper's job is just to launch it under Task
# Scheduler (trigger: at log on) with output captured, and to let Task
# Scheduler's "restart if it fails" setting recover from a crash. State
# (consecutive-loss streak, pending settlements) persists in
# results/btc_5m_live/state.json, so a restart resumes safely rather than
# duplicating or losing track of open positions.

$repoRoot = "D:\Finance\polymarket"
$python = "C:\Users\AlexJ\AppData\Local\Programs\Python\Python312\python.exe"
$logDir = Join-Path $repoRoot "results\btc_5m_live"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$wrapperLog = Join-Path $logDir "wrapper_log.txt"

Set-Location $repoRoot
Add-Content -Path $wrapperLog -Value "`n===== wrapper started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ====="

cmd /c "`"$python`" scripts\run_btc_5m_signal.py 2>&1" | Out-File -Append -FilePath $wrapperLog -Encoding utf8

Add-Content -Path $wrapperLog -Value "===== process exited $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ====="
