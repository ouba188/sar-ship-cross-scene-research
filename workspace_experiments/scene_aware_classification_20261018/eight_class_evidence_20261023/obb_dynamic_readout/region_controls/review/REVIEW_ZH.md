# Region Controls 独立审阅

## 设计判断

当前固定设计无阻塞，待实际脚本与已选预测生成后进行针对性实现审阅和原图回看。本目录不训练、不修改父代码、不重提骨干、不操作 Git。

- shuffled 对每 obs 用 `np.random.default_rng([4711, int(obs_id)]).permutation(3)`，两极化共享排列，整个区域的八类三统计块同时移动；源训练、验证与测试一致。它检验稳定摘要槽位对应关系，不是仅测试时破坏输入。
- common quality 保持实际区域顺序，可能保留恢复区域身份的线索，故不能称完全不可恢复身份的无序对照。
- body/body_end/body_side 在原始 mu/h/b 上清除被排除区域，再进行正常中心化和本折源训练 scaler。69 输入、853 参数相同，但可用特征数不同；不能称有效容量严格相同。
- body 仅限制新增区域摘要，冻结 B 与 common quality 不变，不代表整个系统不看船体外信息。layer1 感受野也不受摘要区域边界约束。
- 动态 full 复用原九个已选头，不能重新选步；新36头同源 J 含0步、200步预算、相同初始化和源训练标准化，源选点全部落盘后固定测试。

## 固定病例规则

使用 seed17 的已选模型。full(dynamic) 对 body 的 rescued/harmed/CC/WW 各取一例，优先至少一个头非零步的组内最小 obs_id；无 active 时保留首例并标记回退。full 对 shuffled 只追加首 rescued/harmed，各最多一例，不按分差挑图。重复对象合并显示，不额外补满，最多六个对象。

救回以比较模型的实际误判类作为固定 rival；误伤和 WW 以 full 实际误判类为 rival；CC 使用 full 最高非真类。另记各模型自己的 top1 margin，不能将固定 rival 分差正数等同预测正确。

复用 native VV/VH、160 crop/pad 与区域覆盖接口，仅重放选定小头。面板显示真实所选分数及端/侧摘要；明确类别中心化、排列或清零前后来源。该轮以训练对照为主，不追加区域删除干预。病例是预定首 ID 说明，不代表总体比例或物理因果。

## 有界补充规则

在原六个比较完成后，实际发现原 OR 活跃条件允许 full=0、对照非零，首个 rescue/harm obs300/1012 均属于此情况。按主流程本次明确要求，保留原四张图和六个比较，不替换原样本；追加 seed17 full vs body 双方 epoch>0 的 rescued/harmed 各最小 obs_id，最多两图。不是按分差幅度选例，不训练、不修改选点、不做新干预。它补充非零头的机制展示，不能把补充病例当成预先冻结的整体证据。完整案例边界见 CASE_REVIEW_ZH.md。

## 实现审阅

已读取实际 `run_controls.py` 与父 `run_heads.py` 的相关接口，未发现需修复的实际错误。

- `run_controls.py:19` 在 summary 的1:4区域轴整体置换，广播维保持两极化同排列、八类与三统计不拆分。删区在原始 summary 清零后调用父 dynamic descriptor，类中心化槽随之清零；common quality 未改。
- `run_controls.py:78` 复用 full 初始化，`83` 仅由 train 拟合 scaler，验证复用；质量列 scaler 与 full 精确比较。`87` 每臂同 seed 初始化、末层零；853名义参数成立。
- `run_controls.py:111` 只在源验证精确 J 严格改善时保存，含0步、每5步、最早平局；test 分支只加载已选 checkpoint/scaler，未用 test 选择。
- 父 `run_heads.py:48` 保留原 train inner / val-test outer 缓存来源，未新拟合 reader。继承既有非完整全栈 OOF 限制，不额外声称无监督历史暴露。
- 病例独立重建 region block descriptor 并与主实现逐值比较，重放选定小头并核对已保存分数；不重提骨干。数值见 replay_checks.csv。
- 既有统计检查文件命名冲突已由主流程修为 `test_pairing_checks`；full分数独立复核另存 `test_full_replay_checks`。本次不改这些父文件。

上述是本轮可达接口的定向审阅，不是重做历史全库审计。
