# ============================================================================
# setup_scheduler.ps1 -- creates the Windows scheduled tasks for LazyStats:
#   LazyStats_ETFDailyStats        15:45 Mon-Fri  daily ETF stats + Telegram
#   LazyStats_AnomalyInvestigation 16:15 Mon-Fri  deterministic anomaly gate;
#                                                  investigates + sends via
#                                                  Telegram only if something
#                                                  qualifies (30 min buffer
#                                                  after ETF stats writes
#                                                  that day's depot row)
#   LazyStats_WeeklyAnomalyReview  10:00 Sat      verifies the week's daily
#                                                  explanations + synthesis,
#                                                  sends via Telegram
#
# (LazyStats_ETFDailyStats/AnomalyInvestigation were previously registered
# ad hoc, outside any version-controlled script -- this brings all three
# under one source of truth.)
#
# Run from PowerShell as administrator:
#     powershell -ExecutionPolicy Bypass -File .\setup_scheduler.ps1
#
# To remove the tasks:
#     powershell -ExecutionPolicy Bypass -File .\setup_scheduler.ps1 -Remove
# ============================================================================
param(
    [switch]$Remove,
    [string]$Root = ""
)

$ErrorActionPreference = "Stop"
if (!$Root) {
    $Root = Split-Path -Parent $MyInvocation.MyCommand.Path
}
$logDir = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$taskNames = @("LazyStats_ETFDailyStats", "LazyStats_AnomalyInvestigation", "LazyStats_WeeklyAnomalyReview")

if ($Remove) {
    foreach ($name in $taskNames) {
        if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) {
            Unregister-ScheduledTask -TaskName $name -Confirm:$false
            Write-Host "Removed task $name"
        }
    }
    Write-Host "Done."
    return
}

function New-LazyStatsTask($name, $wrapperScript, $trigger, $description) {
    $wrapper = Join-Path $Root $wrapperScript
    $logFile = Join-Path $logDir "$name.log"
    # -Command (not -File): Task Scheduler invokes powershell.exe directly, and
    # -File would pass "*>>" through as an inert literal argument instead of
    # redirecting output (same reasoning as the other repos' setup_scheduler.ps1).
    $cmdString = "& '$wrapper' *>> '$logFile'"
    $psArgs = "-NoProfile -ExecutionPolicy Bypass -Command `"$cmdString`""
    $action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $psArgs
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable `
        -DontStopOnIdleEnd -ExecutionTimeLimit (New-TimeSpan -Hours 3)
    if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $name -Confirm:$false
    }
    Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger `
        -Settings $settings -Description $description | Out-Null
    Write-Host "Created task '$name' -> $wrapperScript"
}

New-LazyStatsTask "LazyStats_ETFDailyStats" "run_daily_etf_stats.ps1" `
    (New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At "15:45") `
    "LazyStats: daily ETF returns/volatility/correlation/outliers + Telegram report"

New-LazyStatsTask "LazyStats_AnomalyInvestigation" "run_daily_anomaly_investigation.ps1" `
    (New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At "16:15") `
    "LazyStats: deterministic gate + LLM investigation of flagged statistical anomalies, Telegram only if triggered"

New-LazyStatsTask "LazyStats_WeeklyAnomalyReview" "run_weekly_anomaly_review.ps1" `
    (New-ScheduledTaskTrigger -Weekly -DaysOfWeek Saturday -At "10:00") `
    "LazyStats: Saturday review -- verify the week's anomaly explanations, synthesize trends/regime/risk, Telegram report"

Write-Host ""
Write-Host "Tasks created. Verify with: Get-ScheduledTask -TaskName LazyStats*"
Write-Host "Logs in: $logDir"
