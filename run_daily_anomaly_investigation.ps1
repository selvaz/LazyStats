# LazyStats daily anomaly investigation wrapper -- runs after
# LazyStats_ETFDailyStats (15:45 Pacific) has written that day's
# etf_daily_stats row; the gate reads the two most recent such rows.
# Requires environment variables:
#   LAZYSTATS_RESULT_DEPOT_DB
#   ANOMALY_EXPLANATIONS_DB
#   LAZYCRAWLER_NEWS_DB
# Optional (Telegram send is skipped, not failed, if unset):
#   TELEGRAM_BOT_TOKEN
#   TELEGRAM_CHAT_ID

$ErrorActionPreference = 'Continue'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = 'C:\ProgramData\spyder-6\python.exe'

Set-Location $Root

function Import-PersistedEnvVar($Name) {
    if (Test-Path "Env:$Name") {
        return
    }
    $value = [Environment]::GetEnvironmentVariable($Name, "User")
    if (!$value) {
        $value = [Environment]::GetEnvironmentVariable($Name, "Machine")
    }
    if ($value) {
        Set-Item -Path "Env:$Name" -Value $value
        Write-Host "[$(Get-Date -Format s)] Loaded $Name from persisted environment."
    }
}

Import-PersistedEnvVar "LAZYSTATS_RESULT_DEPOT_DB"
Import-PersistedEnvVar "ANOMALY_EXPLANATIONS_DB"
Import-PersistedEnvVar "LAZYCRAWLER_NEWS_DB"
Import-PersistedEnvVar "TELEGRAM_BOT_TOKEN"
Import-PersistedEnvVar "TELEGRAM_CHAT_ID"

Write-Host "[$(Get-Date -Format s)] Starting daily anomaly investigation"
& $Python (Join-Path $Root 'run_daily_anomaly_investigation.py')
$exitCode = $LASTEXITCODE
Write-Host "[$(Get-Date -Format s)] run_daily_anomaly_investigation.py exit code: $exitCode"

exit $exitCode
