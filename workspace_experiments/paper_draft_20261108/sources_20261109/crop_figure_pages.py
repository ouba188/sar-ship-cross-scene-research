"""Crop slide-page whitespace while preserving native PDF vector content."""
from pathlib import Path
import json
import fitz
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures" / "author_20261109"
source = fitz.open(FIGURES / "paper_flowcharts.pdf")
records = []
for index, page in enumerate(source):
    pix = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
    pixels = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
    yy, xx = np.where(np.any(pixels[:, :, :3] < 245, axis=2))
    clip = fitz.Rect(max(0, xx.min()-6), max(0, yy.min()-6),
                     min(page.rect.width, xx.max()+7),
                     min(page.rect.height, yy.max()+7))
    output = fitz.open()
    target = output.new_page(width=clip.width, height=clip.height)
    target.show_pdf_page(target.rect, source, index, clip=clip)
    stem = f"fig{index+3}_author"
    output.save(FIGURES / f"{stem}.pdf", garbage=4, deflate=True)
    target.get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False).save(FIGURES / f"{stem}.png")
    records.append({"figure": index+3, "slide": index+1, "clip": list(clip),
                    "pdf": f"{stem}.pdf", "png": f"{stem}.png"})
(FIGURES / "figure_manifest.json").write_text(
    json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
print("Prepared five vector PDF figures and matching PNG previews.")
