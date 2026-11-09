$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$out = Join-Path $root 'figures\author_20261109'
$script:editedShapes = @()

function Find-Shape($shapes, [int]$id) {
    foreach ($shape in $shapes) {
        if ($shape.Id -eq $id) { return $shape }
        if ($shape.Type -eq 6) {
            $found = Find-Shape $shape.GroupItems $id
            if ($null -ne $found) { return $found }
        }
    }
    return $null
}

# Native text replaces only the selected legacy Equation objects, in the paper copy.
# Braced subscripts/superscripts remain individually editable PowerPoint text runs.
function Math-Text($slide, $x, $y, $w, $h, [string]$text, $size=11) {
    $shape = $slide.Shapes.AddTextbox(1, $x, $y, $w, $h)
    $shape.Name = 'PaperSymbol_' + $slide.SlideIndex + '_' + $shape.Id
    $shape.Line.Visible = 0
    $shape.Fill.Visible = 0
    $frame = $shape.TextFrame
    $frame.MarginLeft = 0
    $frame.MarginRight = 0
    $frame.MarginTop = 0
    $frame.MarginBottom = 0
    $frame.WordWrap = 0
    $frame.AutoSize = 0
    $shape.TextFrame2.AutoSize = 0
    $frame.VerticalAnchor = 3
    $runs = [regex]::Matches($text, '([_^])\{([^}]+)\}|([^_^]+|[_^])')
    $plain = ''
    $formats = @()
    foreach ($run in $runs) {
        if ($run.Groups[1].Success) {
            $formats += @{ Start=$plain.Length+1; Length=$run.Groups[2].Value.Length; Kind=$run.Groups[1].Value }
            $plain += $run.Groups[2].Value
        } else { $plain += $run.Groups[3].Value }
    }
    $range = $frame.TextRange
    $range.Text = $plain
    $range.Font.Name = 'Times New Roman'
    $range.Font.Size = [single]$size
    $range.ParagraphFormat.Alignment = 2
    foreach ($format in $formats) {
        $font = $range.Characters($format.Start, $format.Length).Font
        $font.Size = [single]($size * 0.7)
        if ($format.Kind -eq '_') { $font.BaselineOffset = [single]-0.25 }
        else { $font.BaselineOffset = [single]0.35 }
    }
    $shape.Width = [single]$w
    $shape.Height = [single]$h
    $shape.Left = [single]$x
    $shape.Top = [single]$y
    $script:editedShapes += $shape
    return $shape
}

function Replace-Math($slide, [int]$id, [string]$text, $size=11, $width=0) {
    $old = Find-Shape $slide.Shapes $id
    if ($null -eq $old) { throw "Missing original symbol $id on slide $($slide.SlideIndex)" }
    $w = $old.Width
    if ($width -gt 0) { $w = $width }
    $shape = Math-Text $slide ($old.Left+($old.Width-$w)/2) $old.Top $w $old.Height $text $size
    $old.Delete()
    return $shape
}

function Edit-Labels($shapes, [int]$slide) {
    for ($n=1; $n -le $shapes.Count; $n++) {
        $shape = $shapes.Item($n)
        if ($shape.Type -eq 6) { Edit-Labels $shape.GroupItems $slide }
        if ($shape.HasTextFrame -eq -1 -and $shape.TextFrame.HasText -eq -1) {
            $originalWidth = $shape.Width
            $originalHeight = $shape.Height
            $range = $shape.TextFrame.TextRange
            $text = $range.Text.Trim()
            if ($text -match 'M[123]') {
                foreach ($pair in @(@('M1','CRFB'), @('M2','RGRP'), @('M3','RCEW'))) {
                    [void]$range.Replace($pair[0], $pair[1], 0, 0, 0)
                }
                if ($text -match '^M[123]') {
                    $range.Font.Size = 6.5
                    $shape.TextFrame.MarginLeft = 0
                    $shape.TextFrame.MarginRight = 0
                }
            }
            if ($slide -eq 1) {
                if ($shape.Id -eq 297) {
                    $range.Text = 'CRFB-guided regional complement'
                    $range.Font.Size = 7
                }
                if ($shape.Id -eq 830) { $range.Text = 'Selection flag' }
                if ($shape.Id -eq 962) { $range.Text = '1x1 Conv' }
                if ($shape.Id -in @(19,28)) { $range.Text = 'RGRP'; $range.Font.Size = 6 }
                if ($shape.Id -eq 13) { $range.Text = 'Pool'; $range.Font.Size = 7 }
                if ($shape.Id -eq 5) { $range.Text = 'Write'; $range.Font.Size = 7 }
                if ($shape.Id -eq 6) { $range.Font.Size = 5.8 }
                if ($shape.Id -eq 504) { $range.Text = 'Shallow RGRP'+[char]13+'Readout'; $range.Font.Size = 8 }
                if ($shape.Id -eq 292) { $range.Text = 'Deep RGRP'+[char]13+'Readout'; $range.Font.Size = 8 }
                if ($shape.Id -eq 46) { $shape.TextFrame.WordWrap = 0 }
            }
            if ($slide -eq 4) {
                if ($text -eq 'Column Marginals') { $range.Text = ('Row'+[char]13+'Marginals') }
                elseif ($text -match '^Row\s+Marginals$') { $range.Text = ('Column'+[char]13+'Marginals') }
                if ($shape.Id -eq 76 -and $text -eq 'CB') { $range.Text = 'SB' }
                if ($shape.Id -eq 44) {
                    $range.Text = 'Outer product / M_s'
                    $range.Font.Size = 5.5
                }
                if ($shape.Id -eq 62) {
                    $range.Text = 'Center(-D/0.1)'
                    $range.Font.Size = 6
                }
            }
            if ($slide -eq 5 -and $text -eq 'tanh') {
                $range.Text = '0.5 tanh'
                $range.Font.Size = 6.5
            }
            if ($slide -eq 5 -and $shape.Id -eq 78) {
                $range.Font.Size = 7.5
                $range.Text = 'Weighted'+[char]13+'Cosine'+[char]13+'Similarity'
                $shape.Left = 440
                $originalWidth = 50
                $originalHeight = 29
                $shape.TextFrame.MarginLeft = 0
                $shape.TextFrame.MarginRight = 0
            }
            if ($shape.Type -eq 17 -and ($range.Text.Trim() -ne $text -or ($slide -eq 1 -and $shape.Id -eq 46))) {
                $shape.TextFrame.AutoSize = 0
                $shape.TextFrame2.AutoSize = 0
                $shape.Width = [single]$originalWidth
                $shape.Height = [single]$originalHeight
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
    $pi = [string][char]0x03C0
    $rho = [string][char]0x03C1
    $kappa = [string][char]0x03BA
    $beta = [string][char]0x03B2
    $delta = [string][char]0x03B4
    $eta = [string][char]0x03B7
    $mu = [string][char]0x03BC
    $neg = [string][char]0x00AC
    $bar = [string][char]0x0304
    $times = [string][char]0x00D7
    $sum = [string][char]0x03A3
    $tilde = [string][char]0x0303
    $neq = [string][char]0x2260
    $lambda = [string][char]0x03BB

    $s = $deck.Slides.Item(2)
    [void](Math-Text $s 260 110 440 18 'CRFB: Class-Response Feedback Block' 12)
    [void](Replace-Math $s 465 "X_{1}(u)^{128${times}20${times}20}" 11 68)
    [void](Replace-Math $s 27 "V_{1}(u)^{32${times}20${times}20}" 11 68)
    [void](Replace-Math $s 28 "A(u)^{K${times}20${times}20}" 11 68)
    [void](Replace-Math $s 62 'A_{1}(u)' 9 30)
    $symbol = Replace-Math $s 100 'A_{2}(u)' 6 20
    $symbol.Left = 187
    [void](Replace-Math $s 63 'A_{K}(u)' 8 30)
    [void](Replace-Math $s 381 "V${bar}_{1}" 9 22)
    [void](Replace-Math $s 365 's_{1}(u)' 8 25)
    [void](Replace-Math $s 367 's_{K}(u)' 8 25)
    [void](Replace-Math $s 377 'C_{k} (32D)' 11 55)
    [void](Replace-Math $s 103 's_{k}(u)' 11 32)
    [void](Replace-Math $s 409 'R_{1}(u)' 11 32)
    $symbol = Replace-Math $s 433 'G_{CRFB}(u) = o(u) tanh(...)' 9 145
    $symbol.Left = 732
    $symbol.Top = 260
    [void](Replace-Math $s 455 'X_{1}(u)' 11 38)
    [void](Replace-Math $s 463 "X'_{1}(u)^{128${times}20${times}20}" 11 72)
    (Find-Shape $s.Shapes 413).Delete()
    (Find-Shape $s.Shapes 411).Delete()
    (Find-Shape $s.Shapes 310).TextFrame.TextRange.Text = 'L'+[char]0x2080
    (Find-Shape $s.Shapes 14).TextFrame.TextRange.Text = 'L'+[char]0x2096
    [void](Replace-Math $s 51 'Class contrast' 6.2 44)
    [void](Math-Text $s 393 295 39 14 'd_{k}(u)' 9)
    [void](Math-Text $s 338 330 148 16 "d_{k}(u) = A_{k}(u) - mean_{j${neq}k} A_{j}(u)" 8)
    [void](Replace-Math $s 110 ('Mean_{k}('+[char]0x2299+')') 7.5 44)
    [void](Label $s 590 132 24 12 ('W'+[char]0x2080) 9)
    [void](Math-Text $s 700 320 18 12 'W_{1}' 8)
    [void](Math-Text $s 790 320 18 12 "${times}0.3" 7)

    $s = $deck.Slides.Item(3)
    [void](Math-Text $s 260 88 470 18 'RGRP: Response-Guided Regional Pooling' 12)
    [void](Replace-Math $s 9 "${pi}_{r}(u)" 21 58)
    [void](Replace-Math $s 45 "${pi}_{r}(u)" 11 33)
    [void](Replace-Math $s 444 'q_{k}(u)' 20 58)
    [void](Replace-Math $s 473 "${delta}^{(2)}_{p,r} (512D)" 13 72)
    $symbol = Replace-Math $s 426 'Concat (160D)' 8 55
    $symbol.Left = 434
    [void](Replace-Math $s 28 "d(u)^{K${times}20${times}20}" 11 62)
    [void](Replace-Math $s 62 'd_{1}(u)' 11 30)
    $symbol = Replace-Math $s 100 'd_{2}(u)' 6 20
    $symbol.Left = 565
    [void](Replace-Math $s 63 'd_{K}(u)' 10 30)
    [void](Replace-Math $s 519 'A(u)' 16 61)

    $s = $deck.Slides.Item(4)
    (Find-Shape $s.Shapes 1430).Delete()
    [void](Math-Text $s 270 151 490 18 'RCEW: Reference-Class Evidence Writeback' 12)
    [void](Replace-Math $s 13 'F_{1}(u)' 12 32)
    [void](Replace-Math $s 21 'k^{q}_{i}, k^{ref}_{sj} (32D)' 7 70)
    [void](Replace-Math $s 1427 'v^{q}_{i}, v^{ref}_{sj} (32D)' 7 70)
    [void](Replace-Math $s 1381 'T_{sij}' 12 43)
    [void](Replace-Math $s 1397 "${rho}_{si}" 13 24)
    [void](Replace-Math $s 1399 "${kappa}_{sj}" 13 24)
    [void](Replace-Math $s 1402 'T^{agg}_{sij}' 12 30)
    [void](Replace-Math $s 1423 "${beta}_{si}" 13 26)
    [void](Replace-Math $s 1415 "v${bar}_{s}" 14 24)
    [void](Replace-Math $s 1434 'S_{sij}' 12 27)
    [void](Replace-Math $s 1444 'E_{c}(u)' 14 40)
    foreach ($id in (1339,1343,1345,1347,1349,1351,1353,1355,1357,1359,1361,1363,1365,1367,1369,1371,1373,1375,1377,1379)) {
        [void](Replace-Math $s $id 'T_{sij}' 7.5 17)
    }
    $symbol = Replace-Math $s 44 'Outer / M_{s}' 6.5
    $symbol.Rotation = 90
    $symbol.Left = [single]($symbol.Left - 2.5)
    [void](Math-Text $s 810 302 95 14 "E_{c}(u) = ${sum}_{i} ${eta}_{i}(u) e${tilde}_{ic}" 7)
    [void](Math-Text $s 810 319 95 12 'F_{3}^{+} = F_{3} + 0.1 W_{p}^{out} E' 6.5)

    $s = $deck.Slides.Item(5)
    [void](Math-Text $s 290 80 340 16 'RCEW: Channel and Spatial Comparison' 10)
    [void](Replace-Math $s 1248 "${mu}_{c}" 11 22)
    [void](Replace-Math $s 1250 "${mu}_{${neg}c}" 11 25)
    [void](Replace-Math $s 71 'w_{c}' 14 26)
    [void](Replace-Math $s 551 'w_{c}' 14 26)
    [void](Replace-Math $s 863 'S_{sij}' 13 28)
    foreach ($id in (743,787,789,791,793,795,797,799,802,804,806,808,810,812,814,816,818,820,822,824)) {
        [void](Replace-Math $s $id 'S_{sij}' 9 18)
    }
    # Correct the response identity and missing projection in the shallow readout.
    $s = $deck.Slides.Item(3)
    [void](Label $s 440 199 65 26 ('Linear 160 to 512'+[char]13+'tanh') 9)
    [void](Label $s 519 185 70 26 '0.1 stopgrad RMS' 9)
    # Both class terms derive from keys; values only enter the aggregation path.
    $s = $deck.Slides.Item(4)
    [void](Label $s 323 203 86 85 '' 8)
    $box = Label $s 332 214 74 74 '' 7
    $box.Line.Visible = -1
    $box.Line.ForeColor.RGB = 8421504
    Arrow $s 365 200 365 214
    Arrow $s 406 220 411 220
    Arrow $s 406 253 411 253
    [void](Math-Text $s 334 217 70 13 'Key similarity S_{sij}' 7)
    [void](Math-Text $s 334 231 70 11 'B_{ic} = max S_{sij} (class c)' 6.3)
    [void](Math-Text $s 334 245 70 13 "b_{ic} = B_{ic} - mean_{c'${neq}c} B_{ic'}" 5.8)
    [void](Math-Text $s 334 260 70 11 'L_{sij} = S_{sij} + b_{ic}' 7)
    [void](Math-Text $s 334 274 70 10 'c = y_{s}' 7)
    # Region sums precede concatenation with the global descriptor.
    $s = $deck.Slides.Item(1)
    [void](Math-Text $s 499 46 58 12 'RGRP' 8)
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
    [void](Replace-Math $s 36 'x_{n}^{0}' 15 25)
    [void](Replace-Math $s 833 'x_{n}^{0}' 15 25)
    [void](Replace-Math $s 831 'z_{n}^{0}' 15 25)
    [void](Replace-Math $s 779 'z_{n}^{S}' 15 25)
    [void](Replace-Math $s 274 'b_{nc}' 14 28)
    [void](Replace-Math $s 793 'u_{n} = 0 or 1' 11 58)
    [void](Replace-Math $s 822 'u_{n} = 1' 12 36)
    [void](Replace-Math $s 808 '0.1 L_{KL,correct,w}' 10 70)
    [void](Replace-Math $s 810 'L_{CE,w}' 13 47)
    [void](Replace-Math $s 276 "${lambda} L_{guide,w}" 12 58)
    [void](Replace-Math $s 806 'L_{adapt}' 10 24)
    [void](Math-Text $s 221 389 40 20 ('Teacher'+[char]13+'correct') 7)

    $audit = foreach ($shape in $script:editedShapes) {
        $range = $shape.TextFrame.TextRange
        $boxWidth = $shape.Width
        $boxHeight = $shape.Height
        if ([math]::Round($shape.Rotation) % 180 -eq 90) {
            $boxWidth = $shape.Height
            $boxHeight = $shape.Width
        }
        [pscustomobject]@{
            Name=$shape.Name; Text=$range.Text
            Width=[math]::Round($boxWidth,2); Height=[math]::Round($boxHeight,2)
            TextWidth=[math]::Round($range.BoundWidth,2); TextHeight=[math]::Round($range.BoundHeight,2)
            Overflow=($range.BoundWidth -gt $boxWidth+0.5 -or $range.BoundHeight -gt $boxHeight+0.5)
        }
    }
    $audit | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath (Join-Path $out 'symbol_layout_audit.json') -Encoding utf8
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
