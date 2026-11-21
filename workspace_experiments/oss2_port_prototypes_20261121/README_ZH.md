# OSS2原图与跨水域类内多样性研究

阅读 `FINDINGS_ZH.md`，图形摘要为 `evidence_summary.svg`（本地另有PNG）。

- `analyze.py`：已有实际源队列、按船只尺寸统计、地理/时相敏感性、实际32参考尺寸覆盖。
- `images.py`：原生TIFF测量、84对同尺寸图像对照、VV/VH共同几何响应、13组本地原图板、缓存源侧错例。
- `extend.py`：排除已知验证/评价身份后的本地GRDH八类描述性元数据扩展。
- `summarize.py`：模型极化可用性、粒度依据、尺寸已覆盖错例、9项针对性检查、摘要图。
- `build_report.py`：将本地报告和原图编成用户专用离线HTML，不上传含原图的HTML或图版到公开仓库。

运行环境：Python + numpy/pandas/scipy/matplotlib/Pillow/tifffile/lxml；HTML编译另需markdown。输入沿用工作区既有OSS2清单/元数据/缓存及本机原始数据路径。此归档不含数据、权重、大缓存及逐船身份清单，因此不能只克隆本文件夹就独立重现数据分析。

`*_local.*`、`RAW_REVIEW_LOCAL.md`及含原图HTML均为本地记录。其他CSV/JSON是本轮公开的聚合结果；不要把历史训练侧正确率当成跨港泛化性能。
