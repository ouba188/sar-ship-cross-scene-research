$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$source = 'C:\Users\25791\WPSDrive\1283162078\WPS云盘\Fig. 1. 24 港数据中三艘船的配对观测.pptx'
$copy = Join-Path $PSScriptRoot 'author_fig1.pptx'
Copy-Item -LiteralPath $source -Destination $copy -Force
$app = New-Object -ComObject PowerPoint.Application
$deck = $null
try {
    $deck = $app.Presentations.Open($copy, -1, 0, 0)
    foreach ($shape in $deck.Slides.Item(1).Shapes) {
        if ($shape.HasTextFrame -and $shape.TextFrame.HasText) {
            [void]$shape.TextFrame.TextRange.Replace('crude oil tanke', 'crude oil tanker')
            [void]$shape.TextFrame.TextRange.Replace('LosAngeles', 'Los Angeles')
        }
    }
    $deck.SaveCopyAs((Join-Path $root 'author_fig1_paper_revision.pptx'), 24)
    $deck.SaveCopyAs((Join-Path $root 'figures\author_20261109\author_fig1_slides.pdf'), 32)
    $deck.Slides.Item(1).Export((Join-Path $root 'figures\author_20261109\author_fig1_full.png'), 'PNG', 3200)
} finally {
    if ($null -ne $deck) { $deck.Close() }
    [void][Runtime.InteropServices.Marshal]::ReleaseComObject($app)
}
