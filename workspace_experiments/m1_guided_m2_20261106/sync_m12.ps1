param([switch]$Copy)
$ErrorActionPreference = 'Stop'
$source = 'E:/临时会话'
$repo = 'E:/临时会话/_sar_ship_cross_scene_incremental_git'
$studies = @(
    'm1_role_separation_20261101',
    'm1_design_position_20261101',
    'm1_tuning_position_20261101',
    'm1_covariance_20261101',
    'm1_coverage_20261101',
    'm1_guided_m2_20261106'
)
$selected = [Collections.Generic.List[IO.FileInfo]]::new()
foreach ($scope in $studies) {
    Get-ChildItem -LiteralPath (Join-Path $source $scope) -File -Recurse | ForEach-Object {
        $relative = [IO.Path]::GetRelativePath((Join-Path $source $scope), $_.FullName)
        $excluded = $relative -match '(^|[\\/])(cache|__pycache__|manifests)([\\/]|$)' -or
                    $_.Name -match '^(manifest|rows|orders)\.json$'
        if (-not $excluded -and $_.Extension -in @('.py','.md','.csv','.json','.html','.png','.ps1') -and $_.Length -lt 2MB) {
            $selected.Add($_)
        }
    }
}
# Include the exact research runtime and imported source dependencies, not weights.
$runtime = Join-Path $source 'sar_m123_standalone/sar_m123'
Get-ChildItem -LiteralPath $runtime -Filter '*.py' -Recurse -File | ForEach-Object {$selected.Add($_)}
$dependencies = @(
    'm3_frequency_increment_20261028/source_hard_task_20261101/oss2/model_task.py',
    'm3_frequency_increment_20261028/source_hard_task_20261101/oss2/review_task.py',
    'm3_frequency_increment_20261028/local_scale_repair_20261101/scaled_operator.py',
    'm3_frequency_increment_20261028/class_competitive_matching_20261031/operator_block.py',
    'm3_frequency_increment_20261028/resolution_matching_20261031/geometry.py',
    'scene_aware_classification_20261018/model_closeout_20261023/final_module_ablations_20261029/model.py'
)
foreach ($relative in $dependencies) {$selected.Add((Get-Item -LiteralPath (Join-Path $source $relative)))}
$changed = @()
foreach ($file in $selected) {
    $relative = [IO.Path]::GetRelativePath($source, $file.FullName)
    $destination = Join-Path $repo (Join-Path 'workspace_experiments' $relative)
    $compress = $file.Extension -eq '.csv' -and $file.Length -gt 1MB
    if ($compress) {$destination += '.gz'}
    $exists = Test-Path -LiteralPath $destination
    if ($exists) {
        $left = [IO.File]::ReadAllBytes($file.FullName)
        if ($compress) {
            $inputStream = [IO.File]::OpenRead($destination)
            $gzip = [IO.Compression.GZipStream]::new($inputStream,[IO.Compression.CompressionMode]::Decompress)
            $buffer = [IO.MemoryStream]::new()
            try {$gzip.CopyTo($buffer); $right=$buffer.ToArray()} finally {$gzip.Dispose();$inputStream.Dispose();$buffer.Dispose()}
        } else {$right = [IO.File]::ReadAllBytes($destination)}
        if ([Linq.Enumerable]::SequenceEqual[byte]($left,$right)) {continue}
    }
    $changed += [pscustomobject]@{Path=$relative; Bytes=$file.Length; Existing=$exists}
    if ($Copy) {
        New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
        if ($compress) {
            $inputStream=[IO.File]::OpenRead($file.FullName)
            $outputStream=[IO.File]::Create($destination)
            $gzip=[IO.Compression.GZipStream]::new($outputStream,[IO.Compression.CompressionLevel]::Optimal)
            try {$inputStream.CopyTo($gzip)} finally {$gzip.Dispose();$outputStream.Dispose();$inputStream.Dispose()}
        } else {Copy-Item -LiteralPath $file.FullName -Destination $destination}
    }
}
$changed | Format-Table -AutoSize
Write-Output ('Selected={0}; Changed={1}; ChangedMiB={2:N2}; Copied={3}' -f $selected.Count,$changed.Count,(($changed|Measure-Object Bytes -Sum).Sum/1MB),$Copy.IsPresent)
