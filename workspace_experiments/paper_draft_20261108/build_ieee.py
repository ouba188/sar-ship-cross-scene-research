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
    wide = number in {2, 3, 4}
    env = "table*" if wide else "table"
    specs = {1: "LCCCC", 2: "LCCC", 3: "LLLC", 4: "LCCCCC",
             5: "LCCC", 6: "LCC", 7: "LCC", 8: "LLCC"}
    out = [rf"\begin{{{env}}}[!t]", r"\centering\footnotesize",
           rf"\setcounter{{table}}{{{number-1}}}",
           r"\caption{" + caption + "}", rf"\label{{tab:{number}}}",
           r"\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.2}",
           r"\begin{tabularx}{\linewidth}{" + specs[number] + "}",
           r"\toprule"]
    for i, row in enumerate(head + rows):
        cells = [convert(cell[4]).replace("\n", " ") for cell in row[1]]
        out.append(" & ".join(cells) + r" \\")
        if i == len(head) - 1:
            out.append(r"\midrule")
    out += [r"\bottomrule", r"\end{tabularx}", rf"\end{{{env}}}"]
    return "\n".join(out)


def figure(number, caption, path=None):
    out = [r"\begin{figure*}[!t]", r"\centering",
           rf"\setcounter{{figure}}{{{number-1}}}"]
    if path:
        height = ".34" if number in {1, 2} else ".21"
        out.append(r"\includegraphics[width=.96\textwidth,height=" +
                   height + r"\textheight,keepaspectratio]{" + path + "}")
    else:
        out.append(r"\fbox{\parbox[c][20mm][c]{.92\textwidth}{\centering " +
                   "流程图预留位置（由作者绘制）" + "}}")
    out += [r"\caption{" + caption + "}", rf"\label{{fig:{number}}}",
            r"\end{figure*}"]
    return "\n".join(out)


def wrap_equations(tex):
    # Break long expressions without changing the controlling mathematical content.
    replacements = {
        1: r"""\begin{aligned}
m_{y,c^\star}(x)&=T_y(x)-T_{c^\star}(x),\\
\Delta m_a&=m_{y,c^\star}(x_a^I)-m_{y,c^\star}(x).
\end{aligned}""",
        2: r"""\begin{aligned}
D_f(I)&=I-G_1(I),\\
D_m(I)&=G_1(I)-G_2(I).
\end{aligned}""",
        6: r"""\begin{aligned}
t_r(F)&=\frac{\sum_u m_r(u)F(u)}{\sum_u m_r(u)},\\
r&\in\{\mathrm{body,end,side}\}.
\end{aligned}""",
        7: r"""\begin{aligned}
F^{+}(u)&=F(u)+\sum_r\widetilde m_r(u)\phi_r(t_r(F)),\\
\widetilde m_r(u)&=\frac{m_r(u)}{\max(1,\sum_jm_j(u))}.
\end{aligned}""",
        9: r"""\begin{aligned}
\delta_{p,r}
&=0.1\,\operatorname{stopgrad}\!\left(
 \operatorname{RMS}(t_r(F_2))\right)\\
&\quad{}\cdot\tanh\!\left(W_p[t_{r,0};t_{r,1};\ldots;t_{r,K}]\right).
\end{aligned}""",
        11: r"""\begin{aligned}
\widetilde T^j_{ab}&=\frac{\rho_a^j\kappa_b^j}{M^j},\\
\bar v^j_a&=\frac{\sum_b\widetilde T^j_{ab}v_{j,b}}{\rho^j_a}.
\end{aligned}""",
        14: r"""\begin{aligned}
E_c(u)&=\sum_a\eta_a(u)\widetilde e_{c,a},\\
F_3^{+}(u)&=F_3(u)+0.1W_p^{\mathrm{out}}E(u).
\end{aligned}""",
        15: r"""\begin{aligned}
z_{p,r}={}&t_r(F_{2,p})+P_p^{\mathrm{deep}}t_r(F_{3,p}^{+})\\
&+\delta_{p,r}.
\end{aligned}""",
        17: r"""\begin{aligned}
\ell_i&=\frac{1}{K-1}\sum_{c\ne y_i}
 \left[b_{i,c}-(z_{i,y_i}-z_{i,c})\right]_+^2,\\
\mathcal L_{\mathrm{floor}}
 &=\frac{\sum_i w_i u_i\ell_i}{\sum_i w_i}.
\end{aligned}""",
        18: r"""\begin{aligned}
\mathcal L_{\mathrm{adapt}}={}&\mathcal L_{\mathrm{CE}}
 +0.1\mathcal L_{\mathrm{KL,correct}}\\
&+\lambda\mathcal L_{\mathrm{floor}}.
\end{aligned}""",
    }
    for n, expression in replacements.items():
        pattern = r"(?<!\\)\\\[(?:(?!\\\]).)*?\\tag\{" + str(n) + r"\}\\\]"
        tex, count = re.subn(pattern,
                            lambda _: "\\[\n" + expression + rf"\tag{{{n}}}" + "\n\\]",
                            tex, flags=re.S)
        if count != 1:
            raise ValueError(f"Equation {n}: expected one display, found {count}")
    return tex


def main():
    global API
    BUILD.mkdir(exist_ok=True)
    proc = subprocess.run([PANDOC, "MANUSCRIPT_ZH.md", "-f", "markdown-implicit_figures", "-t", "json"],
                          cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", check=True)
    document = json.loads(proc.stdout)
    API = document["pandoc-api-version"]
    title = plain({"t": "Para", "c": document["meta"]["title"]["c"]})
    blocks = document["blocks"]
    body = []
    bib = []
    table_caption = None
    section = ""
    for block in blocks:
        kind = block["t"]
        if kind == "Header":
            text = plain({"t": "Para", "c": block["c"][2]})
            section = text
            if text in {"摘要", "Abstract", "参考文献"}:
                continue
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
                bib.append(r"\bibitem{ref" + match[1] + "}" + convert([block]))
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
        if not section:
            english_title = re.sub(r"^\\textbf\{英文题目：\}\s*", "", rendered)
            body.append(r"\title{" + title + r"\\[0.6em]{\large " +
                        english_title + r"}}\author{}\maketitle")
        elif section == "摘要":
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
    content = wrap_equations("\n\n".join(body))
    tex = (ROOT / "ieee_header.tex").read_text(encoding="utf-8") + "\n" + content
    tex += "\n\\IEEEtriggeratref{7}\n\\begin{thebibliography}{99}\n" + "\n\n".join(bib)
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
    shutil.copy2(BUILD / "manuscript_ieee.pdf", ROOT / "manuscript_ieee.pdf")
    print("DELIVERED", ROOT / "manuscript_ieee.pdf", flush=True)


if __name__ == "__main__":
    main()
