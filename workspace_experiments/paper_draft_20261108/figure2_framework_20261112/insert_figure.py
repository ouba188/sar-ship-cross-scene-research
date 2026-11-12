"""One-time insertion of the author's layered-evidence overview."""
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "MANUSCRIPT_ZH.md"
text = source.read_text(encoding="utf-8")
assert "fig2_evidence_framework.jpg" not in text, "Already inserted"
shutil.copy2(source, Path(__file__).parent / "MANUSCRIPT_before.md")

def renumber(match):
    start = int(match[1])
    end = match[3]
    return "图" + str(start + (start >= 2)) + (
        match[2] + str(int(end) + (int(end) >= 2)) if end else "")

text = re.sub(r"图(\d+)(?:([—–至])(\d+))?", renumber, text)
intro = """# 3 问题发现与受控干预证据

为了系统分析跨场景条件下的判别证据，本文构建了一个受观测、干预与替代证据比较启发的分层分析框架。如图2所示，自然观测用于揭示局部响应的场景依赖性，受控干预用于检验局部证据对类别竞争的影响，而替代证据比较进一步分析不同证据配置如何改变分类决策。基于此，本文进一步提出证据引导的学习框架。

![图2 跨场景SAR船舶分类的分层判别证据分析框架。L1展示同船跨场景观测中的极化响应与预测差异；L2展示指定极化、区域和尺度的受控图像变化；L3通过参考证据与回写路径的替代比较，分析分类决策的变化。下方给出证据分析与CRFB、RGRP、RCEW及干预引导训练的设计联系。](figures/fig2_evidence_framework.jpg){width=100%}
"""
text = text.replace("# 3 问题发现与受控干预证据\n", intro, 1)
source.write_text(text, encoding="utf-8")
print("Inserted Figure 2; original Figures 2-12 shifted to 3-13")
