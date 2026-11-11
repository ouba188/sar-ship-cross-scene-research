$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$preview = Join-Path $root 'preview'
New-Item -ItemType Directory -Path $preview -Force | Out-Null
$manifest = Get-Content -LiteralPath (Join-Path $root 'figure_manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$app = New-Object -ComObject PowerPoint.Application
$deck = $null
try {
    $inputDeck = Get-ChildItem -LiteralPath $root -Filter '*.pptx' | Select-Object -First 1
    $deck = $app.Presentations.Open($inputDeck.FullName, -1, 0, 0)
    $deck.SaveCopyAs((Join-Path $root 'figures_preview.pdf'), 32)
    foreach ($entry in $manifest) {
        $deck.Slides.Item([int]$entry.slide).Export((Join-Path $preview ($entry.name + '.png')), 'PNG', 2400, 1350)
    }
    Write-Output ('Exported {0} slides' -f $deck.Slides.Count)
} finally {
    if ($null -ne $deck) { $deck.Close(); [void][Runtime.InteropServices.Marshal]::ReleaseComObject($deck) }
    [void][Runtime.InteropServices.Marshal]::ReleaseComObject($app)
}
