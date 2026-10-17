param(
    [string]$Workspace = 'E:\临时会话',
    [string]$PaperTracking = 'E:\Hermes\paper_tracking',
    [string]$LegacyRepo = 'E:\Hermes\github\cross-port-sar-ship-selection',
    [string]$Destination = $PSScriptRoot
)

$ErrorActionPreference = 'Stop'
$allowed = @('.md', '.py', '.csv', '.json', '.yaml', '.yml', '.txt', '.svg', '.html', '.ipynb', '.geojson', '.jsonl', '.png', '.tex', '.bib', '.js', '.css', '.mjs', '.ps1')
$skipDirectory = '[\\/](__pycache__|\.pytest_cache|cache|tmp|\.git|vendor|node_modules)[\\/]'
$records = [System.Collections.Generic.List[object]]::new()
$skipped = [System.Collections.Generic.List[object]]::new()

function Copy-ResearchFile {
    param([System.IO.FileInfo]$File, [string]$SourceRoot, [string]$TargetRoot, [bool]$Sensitivity)
    $relative = [System.IO.Path]::GetRelativePath($SourceRoot, $File.FullName)
    $reason = $null
    if ($File.FullName -match $skipDirectory) { $reason = 'cache_directory' }
    elseif ($File.FullName -match '[\\/]manual_review[\\/](state|browser_checks)[\\/]') { $reason = 'private_or_test_review_state' }
    elseif ($File.FullName -match '[\\/]manual_review[\\/]lucide\.js$') { $reason = 'vendored_dependency' }
    elseif ($File.Extension -notin $allowed) { $reason = 'binary_or_other_extension' }
    elseif ($Sensitivity -and $File.Extension -eq '.json' -and $File.Name -notmatch 'summary|result|metric|config|manifest') { $reason = 'per_sample_json' }
    elseif ($File.Extension -ne '.csv' -and $File.Length -gt 10MB) { $reason = 'large_non_table' }
    if ($reason) {
        $skipped.Add([pscustomobject]@{scope=$TargetRoot; path=$relative; reason=$reason; bytes=$File.Length})
        return
    }

    $target = Join-Path $TargetRoot $relative
    if ($File.Extension -eq '.csv' -and $File.Length -gt 1MB) { $target += '.gz' }
    $parent = Split-Path -Parent $target
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    if ($target.EndsWith('.gz')) {
        $inputStream = [System.IO.File]::OpenRead($File.FullName)
        try {
            $outputStream = [System.IO.File]::Create($target)
            try {
                $gzip = [System.IO.Compression.GzipStream]::new($outputStream, [System.IO.Compression.CompressionLevel]::Optimal)
                try { $inputStream.CopyTo($gzip) } finally { $gzip.Dispose() }
            } finally { $outputStream.Dispose() }
        } finally { $inputStream.Dispose() }
    } else {
        [System.IO.File]::Copy($File.FullName, $target, $true)
    }
    if ($File.Extension -eq '.csv') {
        $alternate = if ($target.EndsWith('.gz')) { $target.Substring(0, $target.Length - 3) } else { $target + '.gz' }
        if (Test-Path -LiteralPath $alternate) { Remove-Item -LiteralPath $alternate }
    }
    $records.Add([pscustomobject]@{scope=$TargetRoot; path=$relative; source_bytes=$File.Length; stored_bytes=(Get-Item -LiteralPath $target).Length})
}

$experimentDirs = @(Get-ChildItem -LiteralPath $Workspace -Directory | Where-Object {
    $_.Name -match '202610(0[1-9]|1[0-7])'
})
foreach ($dir in $experimentDirs) {
    $targetRoot = Join-Path $Destination 'workspace_experiments' $dir.Name
    foreach ($file in (Get-ChildItem -LiteralPath $dir.FullName -File -Recurse)) {
        Copy-ResearchFile $file $dir.FullName $targetRoot $false
    }
}

$sensitivityRoot = Join-Path $PaperTracking 'sensitivity'
$targetRoot = Join-Path $Destination 'paper_tracking_sensitivity'
foreach ($file in (Get-ChildItem -LiteralPath $sensitivityRoot -File -Recurse | Where-Object { $_.LastWriteTime -ge [datetime]'2026-10-01' })) {
    Copy-ResearchFile $file $sensitivityRoot $targetRoot $true
}

$legacyRoot = Join-Path $LegacyRepo 'paper'
$targetRoot = Join-Path $Destination 'legacy_paper_20261007'
foreach ($file in (Get-ChildItem -LiteralPath $legacyRoot -File -Recurse)) {
    Copy-ResearchFile $file $legacyRoot $targetRoot $false
}

$targetRoot = Join-Path $Destination 'workspace_tools'
foreach ($name in @('sea_ship_redetect_preview.py', 'test_sea_ship_redetect_preview.py')) {
    $path = Join-Path $Workspace $name
    if (Test-Path -LiteralPath $path) {
        Copy-ResearchFile (Get-Item -LiteralPath $path) $Workspace $targetRoot $false
    }
}

$summary = [pscustomobject]@{
    snapshot_date = '2026-10-17'
    snapshot_created_local = (Get-Date -Format 'yyyy-MM-ddTHH:mm:sszzz')
    deferred_studies = @()
    excluded_review_artifacts = @('manual_review/state', 'manual_review/browser_checks', 'manual_review/node_modules', 'manual_review/lucide.js')
    experiment_directories = @($experimentDirs.Name)
    copied_files = $records.Count
    source_bytes = [long](($records | Measure-Object source_bytes -Sum).Sum)
    stored_bytes = [long](($records | Measure-Object stored_bytes -Sum).Sum)
    skipped_files = $skipped.Count
    skipped_by_reason = @($skipped | Group-Object reason | ForEach-Object { [pscustomobject]@{reason=$_.Name; count=$_.Count; bytes=[long](($_.Group | Measure-Object bytes -Sum).Sum)} })
}
$summary | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $Destination 'archive_manifest.json') -Encoding utf8
$summary | Format-List
