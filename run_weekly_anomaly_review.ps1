# LazyStats Saturday weekly anomaly review wrapper.
# LazyStats_ETFDailyStats/anomaly investigation run weekdays only (Mon-Fri),
# so Saturday's review of the week's daily items doesn't need to wait for a
# same-day data refresh -- any reasonable Saturday time is safe.
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

Write-Host "[$(Get-Date -Format s)] Starting weekly anomaly review"
& $Python (Join-Path $Root 'run_weekly_anomaly_review.py')
$exitCode = $LASTEXITCODE
Write-Host "[$(Get-Date -Format s)] run_weekly_anomaly_review.py exit code: $exitCode"

exit $exitCode
