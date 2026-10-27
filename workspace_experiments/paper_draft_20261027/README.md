# 论文初稿：Introduction与Related Work

2026-10-27。先写问题与文献定位，不提前冻结探索中的完整模型。

## 阅读入口

- `index.html`：中文合并稿，含图1与参考文献。
- `introduction_zh.html`：同一中文主稿的兼容入口。
- `ieee_zh/main.pdf`：当前中文版IEEE期刊双栏稿，使用IEEEtran的journal模式、XeLaTeX与xeCJK。
- `ieee_zh/main.tex`：可编辑LaTeX主文件，分节内容由中文Markdown生成；`manuscript_review.pdf`为早期单栏阅读副本，不是当前排版主稿。
- `MANUSCRIPT_ZH.md`：可编辑合并稿。
- `INTRODUCTION_ZH.md`、`RELATED_WORK_ZH.md`：分节源文件。
- `REFERENCES.md`：引用及核实范围；`references.bib`供后续LaTeX使用。
- `EDITOR_NOTES_ZH.md`：写作边界、后续待补内容及受控干预的章节位置。

## 图像

图1为3艘船、6次观测的12张VV/VH原chip，补图为6艘船、12次观测的24张原chip。PNG为300 dpi，PDF保留矢量文字；原始chip不会通过插值或生成变成新增观测细节。`figure_sources.csv`保留原图路径、身份、类别、产品、日期及采样元数据。

这些是24港数据的自然配对案例，不是跨数据集同船证据，也不是独立分离港口/平台/几何因素的实验。原chip本身已是8位显示产品，图中不把灰度差解释为定量后向散射差异。

## 本轮没有写成既定贡献的内容

最终网络、联合条件化M1的效果、最终目标域协议、性能增益和贡献列表均待实验确定。受控干预拟放第3节；引言只提炼问题发现，Related Work定位相关工作而不宣称首次。没有新增训练或改变任何已有模型。

## 复现

本地使用ClearSAR Python环境：

```powershell
& 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe' -B 'E:/临时会话/paper_draft_20261027/build_figures.py'
& 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe' -B 'E:/临时会话/paper_draft_20261027/render_draft.py'
node 'E:/临时会话/paper_draft_20261027/export_review.mjs'
```

绘图依赖NumPy、Pillow、Matplotlib；HTML生成使用Markdown 3.8.2。HTML与早期单栏阅读副本检查复用已有manual_review目录的Playwright及本地Chromium。无需运行Web服务器，直接打开HTML。GitHub归档只收本轮文稿、引用、脚本、来源表和拼图，不包含原始图像数据集或文献全文。

IEEE双栏稿更新：先运行`build_ieee.py`，再执行`ieee_zh/build.ps1`，依次调用XeLaTeX、BibTeX及两次XeLaTeX，不依赖Perl或latexmk。模板使用Letter纸张、默认IEEEtran栏宽与页边距，正文中文宋体、英文Times New Roman，图1跨双栏，参考文献采用IEEEtran数字编号。暂未提供作者、单位和摘要，因此没有虚构这些内容；目前是前两节工作稿，并非完整可投稿版本。具体目标期刊确定后再检查其专用模板要求。
