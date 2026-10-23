# 冻结旧 A+D 的极化读出接入协议

2026-10-23，训练前锁定。仅本目录写入；不改父checkpoint，不Git，不另训C。

## 对象与数据

保留旧 `geometry_evidence_gate_20261021/inputs_f{fold}_s{oldseed}.npz` 中 `base+residual` 的逐样本分数作为唯一基线，源训练取 `held_vessel_gate_20261022/oof_f{fold}.npz` 中同oldseed的 `base+residual`。旧seed0/1/2固定对应新seed17/29/43。

旧 cohort 有15252行，新有效 cohort 有15251行；obs31237已排除。通过obs_id与instance_id核对映射，禁止复用旧整数行号索引新矩。三折train=4392/4419/4418，val=310/286/319，test=1561/1632/1551，完全沿用旧划分及标签。

真实pol矩复用 `calibrated_signed_transfer/signed_features.npz`。训练逐行取新支自身4个inner held对应的moments；旧分数逐行取旧支自身3个inner held预测。各自均排除query MMSI，允许独立交叉拟合的fit集合不同。val/test取outer W moments。组合是身份OOF，不是全链过境OOF；旧OOF允许同过境不同船出现在fit。

## 固定三臂

| 臂 | 输入与读出 | 参数 |
|---|---|---:|
| free_real | 真实VV/VH sigma/kappa，父11维descriptor；复用standard_controls.Head('free_shared')，11→16GELU→1，输出去类公共项 | 209 |
| free_aggregate | 父organized('aggregate_repeated')：两pol先求和，再除有效pol数并重复到有效槽；与real同descriptor/架构/初始化 | 209 |
| score_only | 仅8类中心化旧A+D；复用run_training.Head('score_only')，8→16GELU→8，不接矩或flags | 280 |

前两臂输出是直接共享残差，不乘kappa，不tanh；不称为kappa载体模型。score_only不是同参数量，但作为普通强分数校准参照完整保留。无额外融合、其他头或网格。

## 训练与选择

3fold×3seed×3arms=27轨迹。末层全零，零头严格回到原A+D。descriptor scaler仅拟合源OOF训练输入；free两臂沿父函数在样本与类轴上标准化，score_only在样本轴按8维标准化。

全部200步full-batch AdamW(lr=.003,weight_decay=.01)，jointCE=.5普通CE+.5八类均衡CE。sourceval J=.5OA+.5BA，每5步并含0选点，精确分数比较且tie保留最早。未使用测试选融合、选臂、选参数。27个selected checkpoint与selection_frozen.json全部落盘后，才计算测试结果。

不搬现有C或已训练极化头，不强制改动旧模型OA/BA取舍。这里score_only本身作为直接残差参照训练；不存在先校准C再增量的二级监督。

## 验证与输出

核对标识映射、OOF分配、旧test参考逐分数精确重放、zerohead、selected checkpoint重放、有限输出；free两臂同seed初始化逐参数一致。保存27模型、曲线、选步、逐例分数、成本。

每seed分别报告OA/BA/NLL/Brier及源val；逐fold/类/port/VH/过境分层。real-aggregate及real-score逐例配对，沿父固定预测bootstrap：2000次、seed20261023、pass与MMSI聚类，不把3seed当独立样本，不包含训练选择不确定性。val同时报告915折暴露与去重观测的描述性结果。

每arm的seed17按instance_id排序取rescued/harmed/CC/WW首例，共12面板（若缺翻转明确记载并补CC/WW）。实际查看原VV/VH与160 crop/pad，保留有效flag、原路径、分差及模型实际读出。oldrival=原A+D最高非真类，finalrival=最终最高非真类，另记finalwinner；不得混用。

## 解释边界

此旧A+D不含全部历史目标支持适配，不称历史最终完整系统。新W是独立冻结辅助分类方向，不是旧A+D权重分解。观察到收益/负结果均保留，结合源码、源OOF/outer训练组织及类组成解释；不自动追加网格，不把相关现象写成物理机制证明。
