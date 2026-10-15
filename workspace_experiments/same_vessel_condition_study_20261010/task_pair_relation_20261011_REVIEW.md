# Task-Supervised Support-Pair Modulation 协议短审查

2026-10-11。只读源码与已有缓存；未训练、未评价目标港、未修改主实现。复用 crca-cross-port-experiment 的隔离约定。以下是设计审查，不是尚未完成的新实现验收。

## 结论

现有 native-fold 资源可直接支持这轮训练。建议继续，但完整复用原残差读出；将普通图像证据与角度交互分开比较，不把有符号支持系数解释成相似概率。

## 已核实的资源

- `crossfit_evidence_20261011/source_folds.npz`：1296 张、525 艘源训练船；assignment 对应 fit_obs_ids 的位置，不是全局 obs_id。
- 两种子 31415/27182 的 `correction_episodes_{seed}.npz` 各有 400 个 episode，支持 64、查询 96；保存的是全局 obs_id。实际检查：混 fold 为 0，支持/查询 MMSI 重叠为 0。
- 六个 `fold{k}_{seed}_outputs.npz` 的 excluded_obs_ids 均恰好等于对应 fold；excluded MMSI 与拟合、normalization 文件内 selection_obs_ids 的重叠均为 0。prior/kernel 分别为全局 37893x8/37893x32。
- `crossfit_evidence.py:153` 起逐 fold 重新拟合标准化、PCA 初始化、教师、case/prior；并非在全量拟合后只删除预测行。冻结骨干仍有历史接触边界，不能称整条网络从未见过这些船。

## 具体问题与最小处理

1. **旧增量残差少了 -1，不能照搬。** `condition_support_matching.py:82` 使用 `(2*onehot-prior_s)*present`，原 `paired_evidence_factors.py:174` 使用 `(2*onehot-1-prior_s)*present`。本次已有训练 episode 每个种子有 18 个支持集缺类，因此差异会改变已覆盖类别相对未覆盖类别的分数，并可能改变 CE/预测；不是纯公共偏置。最小修正：新模块统一使用完整原残差 `R=(2*onehot-1-prior_s)*present`，`score=prior_q+(op*multiplier)@R`。零参数应精确返回 baseline；用一个缺类 episode 检查零值和非零增量即可。旧结果保留，不能称此差异已解释先前负结果。

2. **监督的是最终贡献，不是同类/异类符号。** `support_operator` 是含截距的平衡岭回归系数，允许负值；同类支持也可能有负贡献。最小做法：用最终 query 分类损失训练 multiplier，保持原 op/R/缺类掩码，不对 op 做 softmax，不强迫同类边权增加。解释采用 `op_qs * R_s` 对正确类和固定竞争类的贡献变化，同时报救回与误伤。

3. **episode 内必须共用一个 native prior/kernel。** 保存 episode 使用全局 ID，而 assignment 使用源拟合集合位置；直接 assignment[obs_id] 或拼接不同 fold 潜坐标会错。用 obs_id->fold 映射取对应缓存，support/query 同取该 fold。训练归一化只用源训练数据；部署用 full 缓存。native/full 上游训练规模不同，若训练好、留出差，先诊断这种分布差异，不能直接归因角度无效。

4. **普通 evidence、condition、sham 保持同样的样本与训练机会。** 固定相同 episode、base、最大步数、验证时点和选择规则。普通证据版应有可比参数量与非线性容量，而非仅低容量常数；condition 增加角度交互；sham 与 condition 完全同结构。三者使用相同有效边掩码，否则覆盖率与条件信息混杂。原同船波动模型的同港/同平台/同轨限制不必机械继承，但若改为跨船分类学习，应在出结果前明确新适用边界，不按收益放宽。

5. **sham 置乱单位需要与任务一致。** 旧 `incidence_transport.py:37` 在船/港/平台/轨内置乱，单观测或等角组实际上不变。可复用，但须记录实际被改变的训练观测/边比例；若扰动很少，condition≈sham 不能有力否定角度作用。给每个源观测固定置乱角度并在所有 episode 复用，不逐边独立制造彼此矛盾的角度。只置乱训练映射、验证用真实角度属于“破坏训练对应关系”对照，不是完整因果干预；不要用训练船身份去置乱未知查询标签。

6. **新船验证是选择集，不是新确认。** `val_new_ship` 三组支持均覆盖八类，且已用于 upstream checkpoint；可以选新增模块轮次，但不能把它称独立全流程验证。新 OOF 查询标签训练新模块后，也不再是该模块的独立测试。最终源 held 只评价固定选择；不给 held 或目标港反选权重。报告 source-side 八分类 OA/BA，勿写成全 24 港或双骨干主模型提升。

## 与旧 Gate 的实质区别

`conditional_evidence_gate_20261010/FINDINGS_ZH.md` 的旧方案是每图 32->8、264 参数的分类方向增益，同时改变 prior 和支持核；实际主要收益来自 prior 路径。新方案若冻结 prior/kernel，只学习 query-support 配对对最终类别分数的有符号贡献，则研究的是“哪些支持证据在何种条件下值得使用”，而不是旧逐图 gate 改名。无需复跑旧目标港实验；先用本轮同协议普通证据对照判断角度是否提供独特增量。
