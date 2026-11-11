"""Inspect export completeness and make a contact sheet; no model execution."""
from pathlib import Path
import json
import zipfile
import xml.etree.ElementTree as ET
from html import escape
import fitz
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT/'figure_manifest.json').read_text(encoding='utf-8'))
pdf = fitz.open(ROOT/'figures_preview.pdf')
assert len(pdf) == len(manifest) == 9
(ROOT/'pdf').mkdir(exist_ok=True)
sheet = Image.new('RGB', (1440, 870), 'white')
for i, item in enumerate(manifest):
    with Image.open(ROOT/'preview'/f"{item['name']}.png") as image:
        image.thumbnail((480, 270))
        sheet.paste(image, ((i % 3)*480, (i//3)*290))
    ImageDraw.Draw(sheet).text(((i % 3)*480+8,(i//3)*290+271), item['name'], fill='black')
    page = fitz.open()
    page.insert_pdf(pdf, from_page=i, to_page=i)
    page.save(ROOT/'pdf'/f"{item['name']}.pdf")
sheet.save(ROOT/'preview'/'contact_sheet.png')
with zipfile.ZipFile(next(ROOT.glob('*.pptx'))) as archive:
    media = [n for n in archive.namelist() if n.startswith('ppt/media/')]
    assert not media, media
    counts = []
    for i in range(9):
        tree = ET.fromstring(archive.read(f'ppt/slides/slide{i+1}.xml'))
        counts.append(len(tree.findall('.//{http://schemas.openxmlformats.org/presentationml/2006/main}sp')))
report = dict(slides=9, embedded_raster_media=0, editable_shapes_per_slide=counts,
              pdf_pages=len(pdf), previews=9, individual_vector_pdfs=9,
              notes='Native PPTX text/rectangles/lines, not screenshots; visual review required.')
(ROOT/'export_checks.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(report)
cards = []
for item in manifest:
    name = item['name']
    cards.append(f'<section><h2>{escape(name)}</h2><a href="svg/{name}.svg">SVG矢量图</a> · '
                 f'<a href="pdf/{name}.pdf">单图PDF</a><img src="preview/{name}.png" alt="{escape(name)}"></section>')
html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>论文可编辑图表</title>
<style>body{font:16px "Microsoft YaHei",sans-serif;color:#20252b;margin:32px auto;max-width:1200px;padding:0 20px;background:#fff}
h1{font-size:26px}h2{font-size:18px}section{border-top:1px solid #ddd;padding:20px 0}img{width:100%;height:auto;display:block}
a{color:#286a78}p{line-height:1.7}</style><h1>现有实验证据图表</h1>
<p><a href="实验图表_全部可编辑.pptx">可编辑PPTX</a> · <a href="图表数据.xlsx">数据工作簿</a> ·
<a href="figures_preview.pdf">完整PDF</a> · <a href="README_ZH.md">数据范围与指标定义</a></p>
<p>当前完整模型结果、历史受控干预和历史参考预算分开展示；未补造缺失的参考规模点。</p>'''
(ROOT/'index.html').write_text(html+'\n'.join(cards)+'</html>', encoding='utf-8')
