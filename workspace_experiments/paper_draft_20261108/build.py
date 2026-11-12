"""Render the manuscript using archived evidence and author-supplied diagrams."""
from pathlib import Path
import csv
import json
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
FIGURES = {
    "fig2_evidence_framework.jpg": Path("E:/Program_documents/Weixin/xwechat_files/wxid_mjwmzhlgjh6222_f9ca/temp/RWTemp/2026-11/924bcb2c8402d46c5235f48a939e86f7/f65592b940c5f898a413074f64ff266d.jpg"),
    "fig2_intervention_case.png": WORK / "paper_evidence_20261027/full_model_interventions_20261027/figures/case_10488.png",
    "fig6_m1_m2_case.png": WORK / "m1_guided_m2_20261106/mean_rescue_s29_1142.png",
    "fig7_m3_case.png": WORK / "m3_frequency_increment_20261028/source_hard_task_20261101/oss2/internal_contribution/writer_review/evidence_vs_feature_rescue_s17_735.png",
}


def run(args, name):
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    (ROOT / "_build" / (name + ".log")).write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(name + ": " + (result.stdout + result.stderr)[-5000:])
    print(name, "OK", flush=True)


def main():
    (ROOT / "figures").mkdir(exist_ok=True)
    (ROOT / "_build").mkdir(exist_ok=True)
    for name, source in FIGURES.items():
        target = ROOT / "figures" / name
        if not target.exists() and source.exists():
            shutil.copy2(source, target)
        elif not target.exists():
            raise FileNotFoundError(source)
    source = (ROOT / "MANUSCRIPT_ZH.md").read_text(encoding="utf-8")
    manifest = json.loads((ROOT / "figures/author_20261110/manifest.json").read_text(encoding="utf-8"))
    with (ROOT / "figure_sources.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["figure", "archived_file", "source"])
        for number, path in re.findall(r"^!\[图(\d+) [^\n]+\]\(([^)]+)\)", source, re.M):
            origin = manifest[path]["source"] if path in manifest else str(FIGURES[Path(path).name])
            writer.writerow([number, path, origin])
    # Keep the same manual figure numbering as the IEEE edition.
    rendered = re.sub(r"^!\[([^\n]+)\]\(([^)]+)\)(\{[^\n]*\})?$",
                      lambda m: "![](" + m[2] + ")" + (m[3] or "") + "\n\n" + m[1],
                      source, flags=re.M)
    funding = re.search(r'^funding: "(.*)"$', source, re.M).group(1)
    affiliation = re.search(r'^affiliation: "(.*)"$', source, re.M).group(1)
    rendered = rendered.replace("# 摘要", "[^author-details]\n\n# 摘要", 1)
    rendered += "\n\n[^author-details]: " + funding + " " + affiliation + "\n"
    html_rendered = rendered
    figure_numbers = {path: number for number, path in re.findall(
        r"^!\[图(\d+) [^\n]+\]\(([^)]+)\)", source, re.M)}
    for path, info in manifest.items():
        width, height = info["size"]
        x0, y0, x1, y1 = info["viewport"]
        cw, ch = x1 - x0, y1 - y0
        viewport = (f'<div style="position:relative;width:100%;aspect-ratio:{cw}/{ch};overflow:hidden">'
                    f'<img src="{path}" alt="图{figure_numbers.get(path, info["figure"])}" '
                    f'style="position:absolute;max-width:none;width:{100*width/cw:.6f}%;'
                    f'height:auto;left:{-100*x0/cw:.6f}%;top:{-100*y0/ch:.6f}%;margin:0" /></div>')
        html_rendered = re.sub(r"!\[\]\(" + re.escape(path) + r"\)(\{[^\n]*\})?", lambda m: viewport, html_rendered)
        rendered = rendered.replace("(" + path + ")", "(" + str(Path(path).with_suffix(".pdf")).replace("\\", "/") + ")")
    (ROOT / "_build/rendered.md").write_text(rendered, encoding="utf-8")
    (ROOT / "_build/rendered_html.md").write_text(html_rendered, encoding="utf-8")
    pandoc = shutil.which("pandoc")
    if not pandoc:
        raise RuntimeError("Existing Pandoc installation was not found")
    common = [pandoc, "_build/rendered.md", "-f", "markdown-implicit_figures", "--standalone", "--resource-path=."]
    html_args = common.copy()
    html_args[1] = "_build/rendered_html.md"
    run(html_args + ["--mathml", "--toc", "--toc-depth=2", "--css=style.css", "-o", "index.html"], "html")
    latex = common + ["-t", "latex", "-V", "CJKmainfont=SimSun", "-V", "mainfont=Times New Roman",
                      "-V", "sansfont=Arial", "-V", "monofont=Consolas", "-V", "fontsize=11pt",
                      "-V", "geometry:margin=24mm", "-V", "papersize=a4", "-V", "colorlinks=false",
                      "-H", "header.tex", "-o", "_build/manuscript.tex"]
    run(latex, "latex_source")
    tex_path = ROOT / "_build/manuscript.tex"
    tex = tex_path.read_text(encoding="utf-8")
    tex = re.sub(r"\\author\{([^{}]+)\}",
                 lambda m: r"\author{\parbox{.95\textwidth}{\centering " + m[1] + "}}",
                 tex)
    tex_path.write_text(tex, encoding="utf-8")
    engine = shutil.which("xelatex")
    if not engine:
        raise RuntimeError("Existing XeLaTeX installation was not found")
    for number in (1, 2):
        run([engine, "-interaction=nonstopmode", "-halt-on-error", "-output-directory=_build", "_build/manuscript.tex"],
            "xelatex" + str(number))
    shutil.copy2(ROOT / "_build/manuscript.pdf", ROOT / "manuscript_review.pdf")
    print("DELIVERED", ROOT / "manuscript_review.pdf", flush=True)


if __name__ == "__main__":
    main()
