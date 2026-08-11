# LazyStats daily ETF returns/volatility/correlation/outliers wrapper
# Requires environment variables:
#   MARKET_DATA_DB
#   LAZYSTATS_RESULT_DEPOT_DB
# Optional (report send is skipped, not failed, if unset):
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

Import-PersistedEnvVar "MARKET_DATA_DB"
Import-PersistedEnvVar "LAZYSTATS_RESULT_DEPOT_DB"
Import-PersistedEnvVar "TELEGRAM_BOT_TOKEN"
Import-PersistedEnvVar "TELEGRAM_CHAT_ID"

Write-Host "[$(Get-Date -Format s)] Starting daily ETF stats job"
& $Python (Join-Path $Root 'run_daily_etf_stats.py')
$exitCode = $LASTEXITCODE
Write-Host "[$(Get-Date -Format s)] run_daily_etf_stats.py exit code: $exitCode"

exit $exitCode
