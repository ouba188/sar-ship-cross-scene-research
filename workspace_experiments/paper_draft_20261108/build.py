"""Render the manuscript without modifying experiments or drawing diagrams."""
from pathlib import Path
import csv
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
FIGURES = {
    "fig1_natural_pairs.png": WORK / "paper_draft_20261027/figures/fig1_same_vessel_observation_shift.png",
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
        if source.exists():
            shutil.copy2(source, target)
        elif not target.exists():
            raise FileNotFoundError(source)
    with (ROOT / "figure_sources.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["figure", "archived_file", "source"])
        writer.writerows((name.split("_")[0], "figures/" + name, str(source)) for name, source in FIGURES.items())
    source = (ROOT / "MANUSCRIPT_ZH.md").read_text(encoding="utf-8")
    # Keep manual figure numbering: the three author-drawn flowcharts are placeholders.
    rendered = re.sub(r"^!\[([^\n]+)\]\(([^)]+)\)(\{[^\n]*\})?$",
                      lambda m: "![](" + m[2] + ")" + (m[3] or "") + "\n\n" + m[1],
                      source, flags=re.M)
    (ROOT / "_build/rendered.md").write_text(rendered, encoding="utf-8")
    pandoc = shutil.which("pandoc")
    if not pandoc:
        raise RuntimeError("Existing Pandoc installation was not found")
    common = [pandoc, "_build/rendered.md", "-f", "markdown-implicit_figures", "--standalone", "--resource-path=."]
    run(common + ["--mathml", "--toc", "--toc-depth=2", "--css=style.css", "-o", "index.html"], "html")
    latex = common + ["-t", "latex", "-V", "CJKmainfont=SimSun", "-V", "mainfont=Times New Roman",
                      "-V", "sansfont=Arial", "-V", "monofont=Consolas", "-V", "fontsize=11pt",
                      "-V", "geometry:margin=24mm", "-V", "papersize=a4", "-V", "colorlinks=false",
                      "-H", "header.tex", "-o", "_build/manuscript.tex"]
    run(latex, "latex_source")
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
