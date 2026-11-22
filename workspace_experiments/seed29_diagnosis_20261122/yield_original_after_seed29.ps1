$ErrorActionPreference = 'Stop'
$root = 'D:/SAR_Experiments/cross_port_sar_ship_ATR'
$experiment = Join-Path $root 'oss2_three_arm_experiment_20261122'
$statePath = Join-Path $experiment 'state_local.json'
$lockPath = Join-Path $experiment '.run_comparison.lock'
while ($true) {
    $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    if ($state.status -eq 'failed') { throw 'Original run failed; inspect before priority transfer' }
    if ($state.status -eq 'completed_requested_steps' -and -not (Test-Path -LiteralPath $lockPath)) {
        Write-Output 'Original run already completed; no interruption needed'
        exit 0
    }
    if ($state.current_step -eq 'seed43/baseline_stage1') { break }
    if ($state.current_step -like 'seed43/*') { throw 'Missed the first seed43 stage; refusing an unreviewed interruption' }
    Start-Sleep -Seconds 2
}
foreach ($seed in 17,29) {
    foreach ($stage in 'baseline_stage1','baseline_base','baseline_base_validation','disentangle_stage1','disentangle_base','disentangle_base_validation','full_h','full_h_validation') {
        if ($state.steps."seed$seed/$stage".status -ne 'succeeded') { throw 'seed17/29 not fully completed' }
    }
}
$owner = Get-Content -LiteralPath $lockPath -Raw | ConvertFrom-Json
$runner = Get-CimInstance Win32_Process -Filter "ProcessId = $($owner.pid)"
if ($runner.CommandLine -notmatch 'run_comparison.py --train-only') { throw 'Original runner ownership mismatch' }
$children = @(Get-CimInstance Win32_Process -Filter "ParentProcessId = $($owner.pid)")
if ($children.Count -gt 1) { throw 'Unexpected scheduler children' }
foreach ($child in $children) {
    if ($child.CommandLine -notmatch 'sar_port.two_stage.cli' -or $child.CommandLine -notmatch 'seed43.+baseline_stage1.pt') { throw 'Unexpected child command' }
}
$backup = Join-Path $experiment ('interrupted_attempts_local/seed43_priority_yield_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
$resolvedRoot = [IO.Path]::GetFullPath($root).TrimEnd('\') + '\'
foreach ($path in @($backup,(Join-Path $experiment 'runs/seed43'))) {
    if (-not [IO.Path]::GetFullPath($path).StartsWith($resolvedRoot,[StringComparison]::OrdinalIgnoreCase)) { throw 'Path escapes D workspace' }
}
New-Item -ItemType Directory -Path $backup | Out-Null
Copy-Item -LiteralPath $statePath -Destination (Join-Path $backup 'state_before_yield.json')
Copy-Item -LiteralPath $lockPath -Destination (Join-Path $backup 'lock_before_yield.json')
# Stop the scheduler first so a terminated child cannot trigger another stage.
Stop-Process -Id $owner.pid
# The scheduler may have spawned a child between the first inventory and stop.
# Re-enumerate after it cannot launch anything else, and verify each command.
$afterStopChildren = @(Get-CimInstance Win32_Process -Filter "ParentProcessId = $($owner.pid)")
foreach ($child in $afterStopChildren) {
    if ($child.CommandLine -notmatch 'sar_port.two_stage.cli' -or $child.CommandLine -notmatch 'seed43.+baseline_stage1.pt') { throw 'Unexpected child after scheduler exit' }
}
$children = @(@($children) + @($afterStopChildren) | Sort-Object ProcessId -Unique)
foreach ($child in $children) { Stop-Process -Id $child.ProcessId -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 1
if (Get-Process -Id $owner.pid -ErrorAction SilentlyContinue) { throw 'Scheduler did not exit' }
foreach ($child in $children) { if (Get-Process -Id $child.ProcessId -ErrorAction SilentlyContinue) { throw 'Child did not exit' } }
$state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
if ($state.current_step -ne 'seed43/baseline_stage1') { throw 'Unexpected state changed at yield' }
foreach ($name in 'baseline_stage1.pt','baseline_stage1.history.json','baseline_stage1.log') {
    $path = Join-Path $experiment "runs/seed43/$name"
    if (Test-Path -LiteralPath $path) { Move-Item -LiteralPath $path -Destination (Join-Path $backup $name) }
}
$state.steps.PSObject.Properties.Remove('seed43/baseline_stage1')
$state.status = 'yielded_for_seed29_recovery'
$state.current_step = $null
$state.updated_at = [DateTimeOffset]::UtcNow.ToString('o')
$receipt = [pscustomobject]@{at=$state.updated_at;reason='User requested seed29 recovery before seed43';preserved_attempt=$backup;resume='Restart only the interrupted seed43 stage from unchanged initialization/configuration';original_model_and_configs_unchanged=$true}
$state | Add-Member -NotePropertyName priority_yields -NotePropertyValue @($receipt) -Force
$temporary = Join-Path $experiment 'state_priority_yield.tmp'
$state | ConvertTo-Json -Depth 90 | Set-Content -LiteralPath $temporary -Encoding utf8
Move-Item -LiteralPath $temporary -Destination $statePath -Force
if (Test-Path -LiteralPath $lockPath) { Move-Item -LiteralPath $lockPath -Destination (Join-Path $backup 'released_run.lock.json') }
$receipt | ConvertTo-Json
exit 0
