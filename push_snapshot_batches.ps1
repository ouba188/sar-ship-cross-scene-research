param(
    [string]$Source = 'E:\临时会话\_sar_ship_cross_scene_research_git',
    [int]$BatchMiB = 80
)

$ErrorActionPreference = 'Stop'
$env:GIT_TERMINAL_PROMPT = '0'
$repo = $PSScriptRoot
$tracked = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
foreach ($path in (git -C $repo -c core.quotepath=false ls-files)) {
    [void]$tracked.Add($path)
}

$files = @(Get-ChildItem -LiteralPath $Source -File -Recurse -Force | Where-Object {
    [System.IO.Path]::GetRelativePath($Source, $_.FullName) -notmatch '^\.git[\\/]'
} | Sort-Object FullName)
$bytes = 0L
$count = 0
$batch = 0

function Publish-Batch {
    param([int]$FileCount, [long]$FileBytes)
    if ($FileCount -eq 0) { return }
    git -C $repo add -A
    if ($LASTEXITCODE -ne 0) { throw 'git add failed' }
    git -C $repo commit -m "Archive batch $($script:batch + 1): $FileCount files" --quiet
    if ($LASTEXITCODE -ne 0) { throw 'git commit failed' }
    git -C $repo -c http.version=HTTP/1.1 -c http.postBuffer=524288000 push -u origin main
    if ($LASTEXITCODE -ne 0) { throw 'git push failed; rerun this script to retry the committed batch' }
    $script:batch++
    Write-Host "Pushed batch ${script:batch}: $FileCount files, $([math]::Round($FileBytes/1MB,1)) MiB"
}

foreach ($file in $files) {
    $relative = [System.IO.Path]::GetRelativePath($Source, $file.FullName)
    $gitPath = $relative.Replace('\', '/')
    if ($tracked.Contains($gitPath)) { continue }
    $target = Join-Path $repo $relative
    $parent = Split-Path -Parent $target
    if (-not (Test-Path -LiteralPath $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    [System.IO.File]::Copy($file.FullName, $target, $true)
    $bytes += $file.Length
    $count++
    if ($bytes -ge $BatchMiB * 1MB) {
        Publish-Batch $count $bytes
        $bytes = 0L
        $count = 0
    }
}
Publish-Batch $count $bytes
Write-Host "Complete: $($files.Count) source files, remote main at $(git -C $repo rev-parse --short HEAD)"
