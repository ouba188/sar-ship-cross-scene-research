"""Render only the new deck with installed PowerPoint and inspect text bounds."""
from pathlib import Path
import json
import win32com.client
from pptx import Presentation

ROOT = Path(__file__).resolve().parent
PPT = ROOT / 'L3_12457_axis_end_editable.pptx'
OUT = ROOT / 'render'
OUT.mkdir(exist_ok=True)
app = win32com.client.Dispatch('PowerPoint.Application')
deck = app.Presentations.Open(str(PPT), ReadOnly=True, WithWindow=False)
issues = []
counts = []
try:
    for i in range(1, deck.Slides.Count + 1):
        slide = deck.Slides(i)
        slide.Export(str(OUT / f'slide_{i:02d}.png'), 'PNG', 2160, 1200)
        for j in range(1, slide.Shapes.Count + 1):
            sh = slide.Shapes(j)
            if sh.HasTextFrame and sh.TextFrame.HasText:
                tf, tr = sh.TextFrame2, sh.TextFrame2.TextRange
                if tr.BoundHeight > sh.Height - tf.MarginTop - tf.MarginBottom + 2:
                    issues.append(dict(slide=i, shape=sh.Name, text=sh.TextFrame.TextRange.Text,
                                       bound_height=tr.BoundHeight, shape_height=sh.Height))
        counts.append(dict(slide=i, shapes=slide.Shapes.Count))
    deck.SaveAs(str(ROOT / 'L3_12457_axis_end.pdf'), 32)
finally:
    deck.Close()
prs = Presentation(PPT)
for i, slide in enumerate(prs.slides, 1):
    counts[i-1]['textboxes'] = sum(sh.has_text_frame and bool(sh.text) for sh in slide.shapes)
    counts[i-1]['pictures'] = sum(sh.shape_type == 13 for sh in slide.shapes)
    counts[i-1]['heatmap_cells'] = sum('_L2_' in sh.name for sh in slide.shapes)
    for sh in slide.shapes:
        if min(sh.left, sh.top) < -1000 or sh.left + sh.width > prs.slide_width + 1000 or sh.top + sh.height > prs.slide_height + 1000:
            issues.append(dict(slide=i, shape=sh.name, issue='outside_slide'))
report = dict(slides=counts, text_overflow_or_bounds_issues=issues, renderer='Microsoft PowerPoint',
              png_size=[2160, 1200], pdf=str(ROOT / 'L3_12457_axis_end.pdf'))
(ROOT / 'render_check.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)
