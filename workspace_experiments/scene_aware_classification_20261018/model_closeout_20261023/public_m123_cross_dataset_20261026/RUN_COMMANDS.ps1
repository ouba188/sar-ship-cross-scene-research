# Explicitly choose one operation. No training or Git commands.
param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('verify', 'replay-oss1', 'replay-oss2', 'oss1-to-oss2', 'oss2-to-oss1')]
    [string]$Operation,
    [string]$OutputName = ''
)

$ErrorActionPreference = 'Stop'
$pythonExe = 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe'
$branchRoot = $PSScriptRoot
$closeoutRoot = Split-Path -Parent $branchRoot
$oss1Root = Join-Path $closeoutRoot 'oss1_m123_replication_20261026'
$oss2Root = Join-Path $closeoutRoot 'oss2_m123_replication_20261026'
$oss1Geometry = 'E:/临时会话/oss1_joint_geometry_20261017/geometry.csv'
$oss2Geometry = Join-Path $oss2Root 'geometry.csv'

if ($Operation -eq 'verify') {
    & $pythonExe -B (Join-Path $branchRoot 'verify_core.py') --check-cohorts
    exit $LASTEXITCODE
}

$sourceRoot = $oss1Root
$targetRoot = $oss2Root
$sourceM3 = Join-Path $oss1Root 'm3_results_grid01'
$geometry = $oss2Geometry
$extraArgs = @()
$device = 'cuda'
switch ($Operation) {
    'replay-oss1' {
        $targetRoot = $oss1Root
        $geometry = $oss1Geometry
        $device = 'cpu'
        $extraArgs = @('--validate-source-replay')
    }
    'replay-oss2' {
        $sourceRoot = $oss2Root
        $targetRoot = $oss2Root
        $sourceM3 = Join-Path $oss2Root 'm3_results'
        $device = 'cpu'
        $extraArgs = @('--validate-source-replay')
    }
    'oss1-to-oss2' {
        $extraArgs = @('--full-table', (Join-Path $oss2Root 'cross_oss1_to_oss2_table.csv'))
    }
    'oss2-to-oss1' {
        $sourceRoot = $oss2Root
        $targetRoot = $oss1Root
        $sourceM3 = Join-Path $oss2Root 'm3_results'
        $geometry = $oss1Geometry
        $extraArgs = @('--full-table', (Join-Path $oss2Root 'cross_oss2_to_oss1_table.csv'))
    }
}
if (-not $OutputName) { $OutputName = $Operation }
& $pythonExe -B (Join-Path $branchRoot 'transfer.py') `
    --source-root $sourceRoot --source-m3 $sourceM3 --target-root $targetRoot `
    --out (Join-Path $branchRoot $OutputName) `
    --view-inventory (Join-Path $targetRoot 'view_inventory.csv') `
    --geometry $geometry --device $device @extraArgs
exit $LASTEXITCODE
