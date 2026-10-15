# 港口扩展独立数值审计

已完成缓存数值检查：通过。
覆盖 23 港口、345/345 个单元；重算全部九个正式候选的 expert/fusion 共 6210 行。
未运行拟合或训练；未修改主程序、旧规则或旧缓存。

- query/support 使用旧 receipt 指向的全局 manifest 行号，逐个核对旧缓存、港口归属及 support 标签。
- BA 和 macro Brier 按 query 实际出现类别等权，实际类别数为 [3, 4, 6, 7, 8]；Brier 单样本误差仍覆盖全部输出类别。
- paired.csv 不一致行数 0，最大绝对误差 9.38e-13。
- rescue−harm 的 BA/OA 恒等式最大误差分别 4.97e-14/1.42e-14 个百分点。
- perport_summary.csv 先平均同港口 draws，equal_port_summary.csv 再对港口等权；部分完成时仅使用已完成单元，并保留 draws/ports 分母。
- 主表逐港/总表/逐类/查询规模敏感性/指定候选对照的不一致行数：{'port_summary': 0, 'summary': 0, 'classes': 0, 'query_size_sensitivity': 0, 'targeted_summary': 0}。
- 缺失 support 类相对 prior_only 的 logit 最大差 0；对应概率最大变化 0.21358。固定 logit 不能保证 softmax 概率、argmax 或召回不变。
- 独立源码检查：run.py:41/49 将全局 ID 映射到港口特征局部行；run.py:112 按全局 manifest 评价。规则与求解器仅收到 support 标签 sy（52/56/57 行），未发现 query 标签进入规则或参数选择。Features.target() 的旧港分支读取冻结 phi/prior，新港分支由特征与源模型计算；条件特征仅使用 incidence_deg。
- summarize.py 先按 task/budget/layer/arm 平均 draws，再平均港口；逐类全局贡献除全部 23 港口，有 query 的类内召回另算，两者区分正确。

| Budget | Ports | Count−raw OA (pp) | Count−raw BA (pp) | Macro Brier Δ |
|---|---:|---:|---:|---:|
| 16 | 23 | 0.441700 | 0.121888 | 0.00054205 |
| 64 | 23 | 1.871311 | -0.047825 | 0.00113513 |
| 128 | 23 | 1.363957 | -1.095973 | 0.00142298 |

预设 n_query>=30 敏感性：

- B16：22 港，OA +0.331907 pp，BA -0.099844 pp，macro Brier +0.00076134。
- B64：22 港，OA +1.826500 pp，BA -0.277271 pp，macro Brier +0.00168937。
- B128：22 港，OA +1.815565 pp，BA -0.463972 pp，macro Brier +0.00224692。

OA 与 BA 结果混合，不能概括为跨预算全面净收益；B64/B128 的 BA 退化须保留。三预算 macro Brier 均升高（变差）；排除小查询港口后 B16 的 BA 也转负。
此处是固定规则的历史缓存扩展；五个 support draws 不等于五个训练种子，也不是全新独立数据确认。未审计新增事后缓存分数诊断。
