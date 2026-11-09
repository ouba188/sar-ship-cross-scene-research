"""Export the author's first slide with whitespace trimmed, preserving vectors."""
from pathlib import Path
import fitz
import numpy as np

folder = Path(__file__).resolve().parents[1] / "figures/author_20261109"
source = fitz.open(folder / "author_fig1_slides.pdf")
page = source[0]
pix = page.get_pixmap(alpha=False)
pixels = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
yy, xx = np.where(np.any(pixels[:, :, :3] < 245, axis=2))
clip = fitz.Rect(max(0, xx.min()-6), max(0, yy.min()-6),
                 min(page.rect.width, xx.max()+7), min(page.rect.height, yy.max()+7))
output = fitz.open()
target = output.new_page(width=clip.width, height=clip.height)
target.show_pdf_page(target.rect, source, 0, clip=clip)
output.save(folder / "fig1_author.pdf", garbage=4, deflate=True)
target.get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False).save(folder / "fig1_author.png")
print("Figure 1:", clip.width, "x", clip.height, "points; first slide only")
