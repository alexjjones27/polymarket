# Wrapper invoked by Windows Task Scheduler (task name: PolymarketLiveStrategy)
# every 4 hours. Runs the live strategy from this machine's own network,
# since Polymarket geoblocks order placement from GitHub Actions' runner
# IPs (see .github/workflows/polymarket_live_trade.yml, which now only
# scans on a schedule and never attempts --live). Logs everything to
# logs/ (gitignored) and pushes any resulting trade-log/state changes to
# GitHub so the record stays visible regardless of where trading runs.
#
# Deliberately does NOT set $ErrorActionPreference = "Stop" and routes
# every native command through cmd.exe: PowerShell 5.1 wraps a native
# command's stderr output in a terminating NativeCommandError when
# captured directly (e.g. via *>> or 2>&1), which previously made this
# script abort after the first harmless stderr line from py_clob_client_v2
# (a benign "Could not create api key" retry-fallback message) before the
# real strategy logic ever ran. cmd.exe's own redirection avoids that.

$repoRoot = "D:\Finance\polymarket"
$python = "C:\Users\AlexJ\AppData\Local\Programs\Python\Python312\python.exe"
$logDir = Join-Path $repoRoot "logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$logFile = Join-Path $logDir ("live_strategy_{0}.log" -f (Get-Date -Format "yyyy-MM"))

Set-Location $repoRoot
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content -Path $logFile -Value "`n===== $timestamp ====="

cmd /c "`"$python`" scripts\run_live_strategy.py --usd 3 --live 2>&1" | Out-File -Append -FilePath $logFile -Encoding utf8

cmd /c "git add results/polymarket_live_test/trade_log.csv results/polymarket_live_test/open_positions.json 2>&1" | Out-File -Append -FilePath $logFile -Encoding utf8
cmd /c "git diff --cached --quiet"
if ($LASTEXITCODE -ne 0) {
    cmd /c "git commit -m `"Autonomous live strategy run (local scheduler): $timestamp`" 2>&1" | Out-File -Append -FilePath $logFile -Encoding utf8
    cmd /c "git push 2>&1" | Out-File -Append -FilePath $logFile -Encoding utf8
} else {
    Add-Content -Path $logFile -Value "No state changes this run -- nothing to commit."
}
