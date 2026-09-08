# Runs the weather collector a few times daily. Read-only: places no orders.
# Cadence matters because the useful row is the one captured BEFORE resolution --
# a snapshot taken after the day has ended records a market that already knows the
# answer, which is worthless for testing whether the forecast beats the price.
$ErrorActionPreference = "Continue"
$repo = "D:\Finance\polymarket"
Start-Process -FilePath "python" -ArgumentList "$repo\scripts\weather_collect_daily.py" `
    -WorkingDirectory $repo -WindowStyle Hidden
