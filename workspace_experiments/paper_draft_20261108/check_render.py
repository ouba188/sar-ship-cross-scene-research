"""Render review pages and check concrete missing-resource/layout failures."""
from pathlib import Path
import json
import re
import sys
from html.parser import HTMLParser
import fitz
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent


class ImageLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images = []

    def handle_starttag(self, tag, attrs):
        if tag == "img":
            self.images.append(dict(attrs).get("src", ""))


def main():
    ieee = "--ieee" in sys.argv
    qa = ROOT / ("qa_ieee" if ieee else "qa")
    qa.mkdir(exist_ok=True)
    page = ImageLinks()
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    page.feed(html)
    missing = [s for s in page.images if not (ROOT / s).exists()]
    assert not missing, missing
    manuscript = (ROOT / "MANUSCRIPT_ZH.md").read_text(encoding="utf-8")
    equations = [int(n) for n in re.findall(r"\\tag\{(\d+)\}", manuscript)]
    assert len(equations) == len(set(equations)), "Duplicate equation number"
    assert html.count('<math') >= len(equations), "Equation conversion missing"
    document = fitz.open(ROOT / ("manuscript_ieee.pdf" if ieee else "manuscript_review.pdf"))
    all_text = ""
    pages = []
    for i, pdf_page in enumerate(document):
        text = pdf_page.get_text()
        assert text.strip(), f"Empty page {i + 1}"
        all_text += text
        pix = pdf_page.get_pixmap(matrix=fitz.Matrix(1.35, 1.35))
        pix.save(qa / f"page_{i + 1:02d}.png")
        outside = [b[:4] for b in pdf_page.get_text("blocks")
                   if b[0] < 10 or b[2] > pdf_page.rect.width - 10
                   or b[1] < 5 or b[3] > pdf_page.rect.height - 5]
        pages.append(dict(page=i+1, text_characters=len(text), outside=outside))
    for number in equations:
        assert f"({number})" in all_text, f"Missing equation {number}"
    assert not any(p["outside"] for p in pages), pages
    for start in range(0, len(document), 8):
        sheet = Image.new("RGB", (1320, 990), "#dddddd")
        draw = ImageDraw.Draw(sheet)
        for offset in range(min(8, len(document)-start)):
            im = Image.open(qa / f"page_{start + offset + 1:02d}.png")
            im.thumbnail((310, 455))
            x, y = (offset % 4)*330+10, (offset//4)*495+24
            sheet.paste(im, (x, y))
            draw.text((x, y-18), f"Page {start+offset+1}", fill="black")
        sheet.save(qa / f"contact_{start//8+1}.png")
    log_path = "_build_ieee/manuscript_ieee.log" if ieee else "_build/manuscript.log"
    log = (ROOT / log_path).read_text(encoding="utf-8", errors="replace")
    record = dict(pages=len(document), images=len(page.images),
                  equation_count=len(equations), html_math_elements=html.count("<math"),
                  overfull=len(re.findall("Overfull", log)),
                  missing_characters=len(re.findall("Missing character", log)),
                  page_checks=pages)
    (qa / "render_checks.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items() if k != "page_checks"}))


if __name__ == "__main__":
    main()
