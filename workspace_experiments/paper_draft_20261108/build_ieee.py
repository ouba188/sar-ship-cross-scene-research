"""Render the controlling manuscript using IEEEtran journal two-column layout."""
from pathlib import Path
import json
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "_build_ieee"
PANDOC = shutil.which("pandoc")


def convert(blocks):
    doc = {"pandoc-api-version": API, "meta": {}, "blocks": blocks}
    proc = subprocess.run(
        [PANDOC, "-f", "json", "-t", "latex", "--wrap=none"],
        input=json.dumps(doc, ensure_ascii=False), capture_output=True,
        text=True, encoding="utf-8", check=True)
    return proc.stdout.strip()


def plain(block):
    proc = subprocess.run(
        [PANDOC, "-f", "json", "-t", "plain", "--wrap=none"],
        input=json.dumps({"pandoc-api-version": API, "meta": {}, "blocks": [block]},
                         ensure_ascii=False),
        capture_output=True, text=True, encoding="utf-8", check=True)
    return proc.stdout.strip()


def table(block, number, caption):
    c = block["c"]
    head = c[3][1]
    rows = [row for body in c[4] for row in body[3]]
    env = "table*" if number == 5 else "table"
    specs = {1: "LL", 2: "LCCCC", 3: "LCCC", 4: "LLL", 5: "LLLL",
             6: r">{\hsize=1.6\hsize}L>{\hsize=.8\hsize}C>{\hsize=.8\hsize}C>{\hsize=.8\hsize}C",
             7: "LCC", 8: "LCCCCC", 9: "LCC", 10: "LCC",
             11: "LCCC", 12: "LCC", 13: "LCC", 14: "LCCC",
             15: "LCCCC"}
    placement = "!t" if env == "table*" else ("!htbp" if number in {6, 7, 8} else "H")
    out = [rf"\begin{{{env}}}[{placement}]", r"\centering\footnotesize",
           rf"\setcounter{{table}}{{{number-1}}}",
           r"\caption{" + caption + "}", rf"\label{{tab:{number}}}",
           r"\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.2}",
           r"\begin{tabularx}{\linewidth}{" + specs[number] + "}",
           r"\toprule"]
    group = None
    for i, row in enumerate(head + rows):
        cells = [convert(cell[4]).replace("\n", " ") for cell in row[1]]
        if number in {9, 14}:
            label, cells = cells[0], cells[1:]
            if i >= len(head) and label != group:
                if group is not None:
                    out.append(r"\midrule")
                columns = 3 if number == 9 else 4
                out.append(r"\multicolumn{" + str(columns) + r"}{l}{\textit{" + label + r"}} \\")
                group = label
        out.append(" & ".join(cells) + r" \\")
        if i == len(head) - 1:
            out.append(r"\midrule")
    out += [r"\bottomrule", r"\end{tabularx}", rf"\end{{{env}}}"]
    return "\n".join(out)


def figure(number, caption, path=None):
    env = "figure*" if number in {3, 4, 6, 8, 9} else "figure"
    placement = "!htb" if env == "figure" else "!tb"
    out = [rf"\begin{{{env}}}[{placement}]", r"\centering",
           rf"\setcounter{{figure}}{{{number-1}}}"]
    if path:
        if "author_20261109/" in path or "author_20261110/" in path:
            path = str(Path(path).with_suffix(".pdf")).replace("\\", "/")
        height = {1: ".38", 2: ".36", 3: ".42", 4: ".42",
                  5: ".25", 6: ".28", 7: ".30", 8: ".44",
                  9: ".42", 12: ".28"}.get(number, ".28")
        out.append(r"\includegraphics[width=\linewidth,height=" +
                   height + r"\textheight,keepaspectratio]{" + path + "}")
    else:
        out.append(r"\fbox{\parbox[c][20mm][c]{.92\linewidth}{\centering " +
                   "图位预留" + "}}")
    out += [r"\caption{" + caption + "}", rf"\label{{fig:{number}}}",
            rf"\end{{{env}}}"]
    return "\n".join(out)


def main():
    global API
    BUILD.mkdir(exist_ok=True)
    proc = subprocess.run([PANDOC, "MANUSCRIPT_ZH.md", "-f", "markdown-implicit_figures", "-t", "json"],
                          cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", check=True)
    document = json.loads(proc.stdout)
    API = document["pandoc-api-version"]
    title = plain({"t": "Para", "c": document["meta"]["title"]["c"]})
    def metadata(name):
        return convert([{"t": "Plain", "c": document["meta"][name]["c"]}])
    authors = metadata("author").replace("Member, IEEE", r"\IEEEmembership{Member,~IEEE}")
    author_block = (r"\author{" + authors + r"\thanks{" + metadata("funding") +
                    r"}\thanks{" + metadata("affiliation") + "}}")
    blocks = document["blocks"]
    body = [r"\title{" + title + "}" + author_block + r"\maketitle"]
    bib = []
    table_caption = None
    section = ""
    for block in blocks:
        kind = block["t"]
        if kind == "Header":
            text = plain({"t": "Para", "c": block["c"][2]})
            section = text
            if text in {"摘要", "Abstract", "参考文献"}:
                if text == "参考文献":
                    body.append(r"\FloatBarrier")
                continue
            if block["c"][0] == 1 and text.startswith(("5 ", "6 ")):
                body.append(r"\FloatBarrier")
            block["c"][2][0]["c"] = re.sub(
                r"^\d+(?:\.\d+)*\s*", "", block["c"][2][0]["c"])
            if block["c"][2][0]["c"] == "":
                block["c"][2] = block["c"][2][2:]
            body.append(convert([block]))
            continue
        if section == "参考文献":
            match = re.match(r"\[(\d+)\]", block["c"][0].get("c", ""))
            if match:
                block["c"] = block["c"][2:]
                entry = convert([block])
                venue_names = {
                    "IEEE Transactions on Geoscience and Remote Sensing": "IEEE Trans. Geosci. Remote Sens.",
                    "IEEE Transactions on Pattern Analysis and Machine Intelligence": "IEEE Trans. Pattern Anal. Mach. Intell.",
                    "IEEE Transactions on Automation Science and Engineering": "IEEE Trans. Autom. Sci. Eng.",
                    "IEEE Transactions on Circuits and Systems for Video Technology": "IEEE Trans. Circuits Syst. Video Technol.",
                    "IEEE Transactions on Image Processing": "IEEE Trans. Image Process.",
                    "Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition": "Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.",
                    "Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition": "Proc. IEEE Conf. Comput. Vis. Pattern Recognit.",
                    "Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops": "Proc. IEEE/CVF Int. Conf. Comput. Vis. Workshops",
                    "Proceedings of the IEEE/CVF International Conference on Computer Vision": "Proc. IEEE/CVF Int. Conf. Comput. Vis.",
                    "Advances in Neural Information Processing Systems": "Adv. Neural Inf. Process. Syst.",
                    "Proceedings of the AAAI Conference on Artificial Intelligence": "Proc. AAAI Conf. Artif. Intell.",
                }
                for full, short in venue_names.items():
                    entry = entry.replace(full, short)
                bib.append(r"\bibitem{ref" + match[1] + "}" + entry)
            continue
        if kind == "Para" and block["c"] and block["c"][0]["t"] == "Image":
            image = block["c"][0]["c"]
            caption = convert([{"t": "Plain", "c": image[1]}])
            match = re.match(r"图(\d+)\s*(.*)", caption, re.S)
            body.append(figure(int(match[1]), match[2], image[2][0]))
            continue
        if kind == "BlockQuote":
            text = plain(block)
            match = re.match(r"图(\d+)占位[：:](.*)", text, re.S)
            if match:
                body.append(figure(int(match[1]), match[2].strip()))
                continue
        if kind == "Table":
            body.append(table(block, *table_caption))
            table_caption = None
            continue
        rendered = convert([block])
        if section == "摘要":
            if rendered.startswith(r"\textbf{关键词"):
                keywords = re.sub(r"^\\textbf\{关键词：\}\s*", "", rendered)
                body.append(r"\begin{IEEEkeywords}" + keywords + r"\end{IEEEkeywords}")
            else:
                body.append(r"\renewcommand{\abstractname}{摘要}"
                            r"\begin{abstract}" + rendered + r"\end{abstract}")
        elif section == "Abstract":
            body.append(r"\renewcommand{\abstractname}{Abstract}"
                        r"\begin{abstract}" + rendered + r"\end{abstract}")
        else:
            caption = re.match(r"\\textbf\{表(\d+)\s*(.*)\}$", rendered, re.S)
            if caption:
                table_caption = (int(caption[1]), caption[2])
            else:
                body.append(rendered)
    content = "\n\n".join(body)
    tex = (ROOT / "ieee_header.tex").read_text(encoding="utf-8") + "\n" + content
    tex += "\n\\balance\n\\begin{thebibliography}{99}\n" + "\n\n".join(bib)
    tex += "\n\\end{thebibliography}\n\\end{document}\n"
    (ROOT / "manuscript_ieee.tex").write_text(tex, encoding="utf-8")
    engine = shutil.which("xelatex")
    for n in (1, 2):
        result = subprocess.run(
            [engine, "-interaction=nonstopmode", "-halt-on-error",
             "-output-directory=_build_ieee", "manuscript_ieee.tex"],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
            errors="replace")
        (BUILD / f"pass{n}.log").write_text(result.stdout + result.stderr,
                                         encoding="utf-8")
        if result.returncode:
            raise RuntimeError((result.stdout + result.stderr)[-4000:])
        print(f"IEEE pass {n} OK", flush=True)
    output = ROOT / "manuscript_ieee.pdf"
    try:
        shutil.copy2(BUILD / "manuscript_ieee.pdf", output)
    except PermissionError:
        output = ROOT / "manuscript_ieee_20261110_revised.pdf"
        shutil.copy2(BUILD / "manuscript_ieee.pdf", output)
    print("DELIVERED", output, flush=True)


if __name__ == "__main__":
    main()
