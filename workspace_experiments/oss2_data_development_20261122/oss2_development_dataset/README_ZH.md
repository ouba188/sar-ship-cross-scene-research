# OSS2 开发资格草稿

训练 719 / 验证 228 / 历史测试回归 378；另隔离 194 条历史 selection。对应四个 `*_qualification_draft_local.json`。

训练记录已恢复 48 路缺检测极化分支；其中 3 条有局部审核通过的 Q，1 条保存候选 Q。原始清单、图像和既有 split 字段保留；新的用途读 `development_split`。

全部 `training_qualified=false`、`leakage_controlled=false`。原生物理有效区尚待补证，验证/测试 606 条配准来源链也未完成。本目录不是训练就绪的数据包；当前模型 loader 不自动强制这些资格字段，不能把该草稿直接传入正式训练入口。

原生输入采用既定 64/96/128 crop/pad 桶；清单中旧 `canvas_size=160` 仅是保留的历史元数据，不能用来覆盖新输入策略。本轮展示保持原始 chip 尺寸，未执行模型预处理或前向。

参见上级 `OSS2_DEVELOPMENT_README_ZH.md` 与 `OSS2_Q_GALLERY_LOCAL.md`。生成脚本 `assemble_oss2_development.py`，不调用训练。
