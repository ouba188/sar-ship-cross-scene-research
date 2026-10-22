# 固定 jointCE 与源 J 选择的匹配对照

2026-10-23，运行前冻结。主候选是父目录已经完成的 signed-only jointCE+J；不重训候选，不以测试更高的 NLL 单元替代它。

只运行三个授权对照：

1. `score_only`：中心化三维 A 类别分数，经 source OOF 每个类轴自己的均值/标准差，3->16 GELU->3 普通 logits residual；115 参数，末层零初始化，不设 tanh 幅度上限，不用局部矩。不要求类别置换等变，因为它是普通类别校准对照。
2. `conditional_std`：直接复用父 81 参数 Correction；输入中心化 A、sigma、类中心化 sigma（相同共享标准化），无 kappa，tanh 系数乘 sigma、输出减类均值。
3. `linear_distribution`：直接复用父 3 参数共享线性头；输入中心化 A、类中心化 sigma/kappa，无 bias，零初始化，确定性 seed0 一次（记录 seed=-1）。

共同规则：原六折，source OOF base/对应 inner 矩，outer validation/test，数据与候选完全相同；训练 jointCE=.5 ordinary CE+.5 balanced CE，AdamW .003/.01、200 步、每 5 步检查、两种 MLP seed17/29/43。42 条轨迹（6*(3+3+1)），每轨迹保存两个选点，共 84 个 selected outputs。

主选择 J=.5 validation OA+.5 validation BA（比例），含 epoch0，严格有理数比较、平局最早、无 NLL 次级排序。balanced NLL 选点仅次要描述，规则沿父改善 >1e-8。全部选点保存后统一评价测试一次；不调参数，不改变候选。score-only 的每类 scaler 仅拟合源 OOF train，其他两臂沿父共享 descriptor scaler，不能将其解释为完全相同归纳偏置。

保存曲线/权重/预测/逐折与 pooled/class/pass/成本/replay。主比较是固定 signed-only jointCE+J 对三个 jointCE+J 对照，按 seed 配对；确定性线性头对各 seed 使用相同预测，不伪造三个训练 seed。次要 NLL 结果单独报告，不选最好测试单元。

原图固定 seed17（确定性 linear 为 -1），每个主对照及已缓存主候选取救回/误伤/CC/WW 首 ID，去重后同图比较；没有某种结果如实记录。原 native VV/VH、尺寸、OBB 与非重叠文字；不增加原图提取或全量空间 map。来源队列为开发数据，不是独立验证。只写 matched_controls，父脚本与候选不动，无 Git。
