$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$out = Join-Path $root 'figures\author_20261109'

function Edit-Labels($shapes, [int]$slide) {
    for ($n=1; $n -le $shapes.Count; $n++) {
        $shape = $shapes.Item($n)
        if ($shape.Type -eq 6) { Edit-Labels $shape.GroupItems $slide }
        if ($shape.HasTextFrame -eq -1 -and $shape.TextFrame.HasText -eq -1) {
            $range = $shape.TextFrame.TextRange
            $text = $range.Text.Trim()
            if ($slide -eq 4) {
                if ($text -eq 'Column Marginals') { $range.Text = ('Row'+[char]13+'Marginals') }
                elseif ($text -match '^Row\s+Marginals$') { $range.Text = ('Column'+[char]13+'Marginals') }
                if ($shape.Id -eq 76 -and $text -eq 'CB') { $range.Text = 'SB' }
                if ($shape.Id -eq 44) {
                    $range.Text = 'Outer product'
                    $range.Font.Size = 6
                }
                if ($shape.Id -eq 62) {
                    $range.Text = 'Center(-D/0.1)'
                    $range.Font.Size = 6
                }
            }
            if ($slide -eq 5 -and $text -eq 'tanh') {
                $range.Text = '0.5 tanh'
                $range.Font.Size = 9
            }
        }
    }
}

function Label($slide, $x, $y, $w, $h, $text, $size=9) {
    $cover = $slide.Shapes.AddShape(1, $x, $y, $w, $h)
    $cover.Fill.Solid()
    $cover.Fill.ForeColor.RGB = 16777215
    $cover.Fill.Transparency = 0
    $cover.Line.Visible = 0
    $s = $slide.Shapes.AddTextbox(1, $x, $y, $w, $h)
    $s.Fill.Visible = -1
    $s.Fill.ForeColor.RGB = 16777215
    $s.Line.Visible = 0
    $s.TextFrame.MarginLeft = 0
    $s.TextFrame.MarginRight = 0
    $s.TextFrame.MarginTop = 0
    $s.TextFrame.MarginBottom = 0
    $s.TextFrame.VerticalAnchor = 3
    $s.TextFrame.TextRange.Text = $text
    $s.TextFrame.TextRange.Font.Name = 'Times New Roman'
    $s.TextFrame.TextRange.Font.Size = $size
    $s.TextFrame.TextRange.ParagraphFormat.Alignment = 2
    $s.TextFrame.AutoSize = 0
    $s.TextFrame2.AutoSize = 0
    $s.Width = $w
    $s.Height = $h
    return $s
}

function Arrow($slide, $x1, $y1, $x2, $y2) {
    $line = $slide.Shapes.AddLine($x1,$y1,$x2,$y2)
    $line.Line.ForeColor.RGB = 0
    $line.Line.Weight = 0.7
    $line.Line.EndArrowheadStyle = 3
}

$app = New-Object -ComObject PowerPoint.Application
$deck = $null
try {
    $deck = $app.Presentations.Open((Join-Path $PSScriptRoot 'author_flowcharts.pptx'), -1, 0, 0)
    for ($i=1; $i -le 5; $i++) { Edit-Labels $deck.Slides.Item($i).Shapes $i }
    # Correct the response identity and missing projection in the shallow readout.
    $s = $deck.Slides.Item(3)
    [void](Label $s 752 390 61 25 'A(u)' 16)
    [void](Label $s 440 199 65 26 ('Linear 160 to 512'+[char]13+'tanh') 9)
    [void](Label $s 519 185 70 26 '0.1 stopgrad RMS' 9)
    # Both class terms derive from keys; values only enter the aggregation path.
    $s = $deck.Slides.Item(4)
    [void](Label $s 323 203 86 85 '' 8)
    $box = Label $s 332 214 74 74 ('Key similarity S'+[char]13+'Class contrast'+[char]13+'b = B_c - mean(B_other)'+[char]13+'L = S + b') 7
    $box.Line.Visible = -1
    $box.Line.ForeColor.RGB = 8421504
    Arrow $s 365 200 365 214
    Arrow $s 406 220 411 220
    Arrow $s 406 253 411 253
    # Region sums precede concatenation with the global descriptor.
    $s = $deck.Slides.Item(1)
    [void](Label $s 747 37 132 76 '' 8)
    [void](Label $s 772 48 63 21 ('Region sum'+[char]13+'(512 per region)') 8)
    [void](Label $s 808 78 30 14 'Concat' 8)
    [void](Label $s 849 77 25 16 'CLS' 10)
    Arrow $s 821 37 821 48
    Arrow $s 747 30 782 48
    Arrow $s 747 104 781 69
    Arrow $s 804 69 823 78
    Arrow $s 747 73 760 85
    Arrow $s 760 85 808 85
    Arrow $s 838 85 849 85
    Arrow $s 874 85 881 85
    [void](Label $s 750 88 48 10 'Global (2048)' 7)
    $deck.SaveCopyAs((Join-Path $root 'author_flowcharts_paper_revision.pptx'), 24)
    $deck.SaveCopyAs((Join-Path $out 'paper_flowcharts.pdf'), 32)
    for ($i=1; $i -le 5; $i++) {
        $deck.Slides.Item($i).Export((Join-Path $out ('paper_slide_{0:00}.png' -f $i)), 'PNG', 3200)
    }
    Write-Output 'Exported five corrected paper-copy diagrams; original source unchanged.'
} finally {
    if ($null -ne $deck) { $deck.Close() }
    [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($app)
}
