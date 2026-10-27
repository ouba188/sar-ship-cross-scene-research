"""Build the reading copy from the editable manuscript and checked references."""
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import markdown

ROOT = Path(__file__).resolve().parent
TITLE = "成像条件与场景变化下的SAR船舶分类"
CSS = """
:root{color:#252a2b;background:#f3f5f5;font-family:Georgia,'Times New Roman','Microsoft YaHei',serif;letter-spacing:0}
*{box-sizing:border-box}body{margin:0}a{color:#17665e;text-underline-offset:3px}nav{font-family:Arial,'Microsoft YaHei',sans-serif;display:flex;gap:20px;flex-wrap:wrap;border-bottom:1px solid #d5ddda;padding:15px max(24px,calc((100% - 1000px)/2));background:#fff;font-size:13px}
header,main{max-width:1060px;margin:auto;background:white;padding:28px 42px}header{padding-bottom:10px}.edition{font:12px Arial,'Microsoft YaHei',sans-serif;color:#687270;margin:0 0 12px}header h1{font-size:28px;line-height:1.3;font-weight:600;margin:0;max-width:850px}header .status{font:13px/1.65 Arial,'Microsoft YaHei',sans-serif;color:#59645f;margin:16px 0 0;border-left:3px solid #39867c;padding-left:12px}
main{padding-top:12px;padding-bottom:60px}article{font-size:17px;line-height:1.75}article h1{font-size:23px;line-height:1.3;margin:28px 0 16px}h2{font-size:18px;line-height:1.5;margin:26px 0 12px}h3{font-size:16px}p{margin:0 0 17px;overflow-wrap:break-word}article>p,article>h1,article>h2,article>h3,article>ul,article>ol{max-width:80ch;margin-left:auto;margin-right:auto}figure{margin:24px 0 30px;break-inside:avoid}figure img{width:100%;height:auto;display:block}figcaption{font-size:13px;line-height:1.6;margin:12px 0 0;color:#414849}figure p{margin:0}table{border-collapse:collapse;font-size:14px;width:100%}td,th{border-bottom:1px solid #cdd4d1;text-align:left;padding:8px}code{font-size:.82em;overflow-wrap:anywhere}.references{font-size:13px;line-height:1.6}.references p{margin-bottom:12px}.citation{font-size:.88em;white-space:nowrap}footer{max-width:1060px;padding:20px 42px;margin:auto;color:#64716c;font:12px/1.6 Arial,sans-serif}.auxiliary{margin-top:35px;border-top:1px solid #ccd4d1;padding-top:20px;font:14px/1.7 Arial,'Microsoft YaHei',sans-serif}
@media(max-width:700px){header,main{padding:22px 18px}header h1{font-size:23px}article{font-size:16px}nav{padding:14px 18px;gap:15px}figcaption{font-size:12px}}
@page{size:A4;margin:18mm 17mm 18mm}
@media print{:root{background:white;color:black}nav,footer,.status,.auxiliary{display:none}header,main{max-width:none;padding:0}header h1{font-size:19pt}.edition{font-size:8pt;margin-bottom:6mm}main{padding-top:4mm}article{font-size:11pt;line-height:1.45}article h1{font-size:14pt;margin-top:7mm}h2{font-size:12pt;break-after:avoid}article p{orphans:3;widows:3}figure{margin:5mm 0;break-inside:avoid}figure img{max-height:155mm;object-fit:contain}figcaption{font-size:8pt;line-height:1.35}.references{font-size:8.5pt;line-height:1.35}a{color:black;text-decoration:none}}
"""


def body(text):
    result = markdown.markdown(text, extensions=["tables", "fenced_code"])
    tree = ET.fromstring("<article>" + result + "</article>")
    children = list(tree)
    for child in children:
        if child.tag == "p" and len(child) == 1 and child[0].tag == "img":
            index = list(tree).index(child)
            figure = ET.Element("figure")
            figure.append(child[0])
            tree.remove(child); tree.insert(index, figure)
            current = list(tree)
            if index + 1 < len(current) and current[index + 1].tag == "p":
                caption = current[index + 1]
                if "Fig. 1." in "".join(caption.itertext())[:30] or "图1." in "".join(caption.itertext())[:30]:
                    caption.tag = "figcaption"; tree.remove(caption); figure.append(caption)
    result = ET.tostring(tree, encoding="unicode", method="html")
    return re.sub(r"\[R(\d+)\]", r'<a class="citation" href="#ref-R\1">[R\1]</a>', result)


def reference_entries(text):
    entries = []
    for block in re.split(r"(?m)^## ", text)[1:]:
        first_line, _, content = block.partition("\n")
        number = re.search(r"\[R(\d+)\]", first_line)
        if number:
            paragraph = content.strip().split("\n\n", 1)[0]
            entries.append((number.group(1), paragraph))
    if not entries:
        raise ValueError("No numbered verified references found")
    return entries


def page(content, language="zh-CN", title=TITLE, status="中文初稿：Introduction与Related Work。完整模型、方法概述及最终贡献尚未定稿。"):
    return f'''<!doctype html><html lang="{language}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><style>{CSS}</style></head><body>
<nav><a href="index.html">中文主稿</a><a href="ieee_zh/main.pdf">IEEE双栏PDF</a><a href="evidence_notes.html">证据与写作边界</a><a href="reference_audit.html">引用核查</a></nav>
<header><p class="edition">论文工作稿 · 2026年10月27日</p><h1>{title}</h1><p class="status">{status}</p></header><main>{content}</main><footer>本轮仅写问题与文献定位，不代表最终模型或性能结论。</footer></body></html>'''


def main():
    intro = (ROOT / "INTRODUCTION_ZH.md").read_text(encoding="utf-8")
    related = (ROOT / "RELATED_WORK_ZH.md").read_text(encoding="utf-8")
    references = (ROOT / "REFERENCES.md").read_text(encoding="utf-8")
    entries = reference_entries(references)
    text = intro + "\n\n" + related
    unresolved = re.findall(r"\[(?:DATASET|POLAR|PERTURB)_REFS\]", text)
    if unresolved:
        raise ValueError(f"Unresolved citation placeholders: {unresolved}")
    used = set(re.findall(r"\[R(\d+)\]", text))
    assert used <= {n for n, _ in entries}, "Missing cited bibliography entry"
    ref_html = '<section class="references"><h1>参考文献</h1>' + "".join(
        f'<div id="ref-R{n}"><p>[R{n}] ' + markdown.markdown(p).removeprefix("<p>").removesuffix("</p>") + '</p></div>'
        for n, p in entries) + '</section>'
    auxiliary = '<section class="auxiliary"><h2>补充观测图</h2><p><a href="figures/figS1_extended_observations.png">6艘船的配对观测，共24张原始chip</a> · <a href="figures/fig1_same_vessel_observation_shift.pdf">图1 PDF</a> · <a href="figure_sources.csv">图像来源表</a></p></section>'
    (ROOT / "index.html").write_text(page(body(text) + ref_html + auxiliary), encoding="utf-8")
    manuscript = "# " + TITLE + "\n\n" + text + "\n\n# 参考文献\n\n" + "\n\n".join(f"[R{n}] {p}" for n, p in entries)
    (ROOT / "MANUSCRIPT_ZH.md").write_text(manuscript, encoding="utf-8")
    (ROOT / "introduction_zh.html").write_text(page(body(text) + ref_html), encoding="utf-8")
    (ROOT / "reference_audit.html").write_text(page(body(references), "zh-CN", "引用核查记录", "仅支持已核实的具体主张；核查范围见各条目。"), encoding="utf-8")
    notes = (ROOT / "EDITOR_NOTES_ZH.md").read_text(encoding="utf-8")
    (ROOT / "evidence_notes.html").write_text(page(body(notes), "zh-CN", "证据与写作边界", "本页是编辑记录，不属于论文正文。"), encoding="utf-8")
    print(f"Rendered reading copy; {len(entries)} verified bibliography entries; {len(used)} cited entries.")


if __name__ == "__main__":
    main()
