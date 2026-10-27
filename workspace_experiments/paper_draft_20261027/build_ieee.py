"""Generate the IEEE journal source from the controlling Chinese Markdown."""
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "ieee_zh"
KEYS = ["huang2018opensarship", "hou2020fusar", "zhao2022dsan",
        "xie2023hogsiam", "shao2023dpig", "he2024cpinet",
        "li2023noncausality", "fong2017meaningful", "hooker2019roar",
        "yin2019fourier", "ross2017right"]


def inline(text):
    text = text.replace("&", r"\&").replace("%", r"\%")
    text = re.sub(r"\*\*(.*?)\*\*", lambda m: r"\textbf{" + m[1] + "}", text)
    text = re.sub(r"(?:\[R\d+\])+", lambda m: r"\cite{" + ",".join(
        KEYS[int(n) - 1] for n in re.findall(r"\d+", m[0])) + "}", text)
    text = text.replace("图1展示", r"图\ref{fig:observations}展示")
    # The empirical section is not yet part of this two-section draft.
    text = text.replace("第3节", "后续问题分析部分").replace("第2节", r"第\ref{sec:related}节")
    return text


def convert(source, output, section, label):
    lines = source.read_text(encoding="utf-8").splitlines()
    result = [r"\section{" + section + "}", r"\label{" + label + "}"]
    for line in lines:
        if line.startswith("# ") or line.startswith("!["):
            continue
        if line.startswith("## "):
            heading = re.sub(r"^## \d+\.\d+\s+", "", line)
            result.append(r"\subsection{" + inline(heading) + "}")
        elif line.startswith("**图1.**"):
            caption = inline(line.removeprefix("**图1.** "))
            result.extend([r"\begin{figure*}[!t]", r"\centering",
                           r"\includegraphics[width=\textwidth]{fig1_same_vessel_observation_shift.png}",
                           r"\caption{" + caption + "}",
                           r"\label{fig:observations}", r"\end{figure*}"])
        else:
            result.append(inline(line))
    output.write_text("\n".join(result) + "\n", encoding="utf-8")


if __name__ == "__main__":
    (OUT / "figures").mkdir(exist_ok=True)
    convert(ROOT / "INTRODUCTION_ZH.md", OUT / "introduction.tex", "Introduction", "sec:intro")
    convert(ROOT / "RELATED_WORK_ZH.md", OUT / "related_work.tex", "Related Work", "sec:related")
    shutil.copy2(ROOT / "references.bib", OUT / "references.bib")
    shutil.copy2(ROOT / "figures/fig1_same_vessel_observation_shift.png", OUT / "figures/fig1_same_vessel_observation_shift.png")
    print(f"Generated IEEE journal inputs in {OUT}")
