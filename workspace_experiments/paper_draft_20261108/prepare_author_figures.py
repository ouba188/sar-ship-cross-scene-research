"""Archive author PNGs unchanged and set document viewports around the artwork."""
from pathlib import Path
import argparse
import json
import shutil

import fitz
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures/author_20261110"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    flows = sorted((args.source / "跨场景SAR分类").glob("*.png"))
    cases = sorted((args.source / "Fig. 1. 24 港数据中三艘船的配对观测").glob("*.png"))
    maps = list((args.source / "地图分布").glob("*.png"))
    if (len(flows), len(cases), len(maps)) != (5, 4, 1):
        raise ValueError("Expected five flowcharts, four case panels, and one map")
    inputs = {1: cases[0], 2: cases[1], 8: maps[0], 9: cases[2], 12: cases[3]}
    inputs.update({i + 3: path for i, path in enumerate(flows)})
    manifest = {}
    for number, source in sorted(inputs.items()):
        target = OUT / f"fig{number}_author.png"
        shutil.copy2(source, target)
        with Image.open(target) as original:
            rgb = original.convert("RGB")
            difference = ImageChops.difference(rgb, Image.new("RGB", rgb.size, "white"))
            visible = difference.convert("L").point(lambda p: 255 if p > 16 else 0)
            bounds = visible.getbbox()
            if bounds is None:
                raise ValueError(f"Empty artwork: {source.name}")
            width, height = rgb.size
            x0, y0, x1, y1 = bounds
            crop = [max(0, x0 - 24), max(0, y0 - 24),
                    min(width, x1 + 24), min(height, y1 + 24)]
        # Crop the PDF page viewport, never resample or paint the supplied PNG.
        document = fitz.open()
        page = document.new_page(width=width, height=height)
        page.insert_image(page.rect, filename=str(target))
        page.set_cropbox(fitz.Rect(crop))
        document.save(target.with_suffix(".pdf"), deflate=True)
        document.close()
        key = target.relative_to(ROOT).as_posix()
        manifest[key] = {"figure": number, "source": source.relative_to(args.source).as_posix(),
                         "size": [width, height], "viewport": crop,
                         "pixels": "author PNG copied unchanged"}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
