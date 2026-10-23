# 区域配套训练对照：有界原图病例

## 结论

实现定向审阅未发现实际阻塞，见 REVIEW_ZH.md。已保留原六个比较、四个对象，再按追加规则补双方非零头的两个首 ID，共八个比较、六个唯一 obs，六张面板均由本 agent 实际 `view_image` 查看。没有训练、重提骨干、区域删除或额外选点；固定小头分数重放最大误差 `4.172325134277344e-07`。

本轮支持的是特定源选择协议下的 BA 取舍，不是 full 全面优于 body/shuffled。尤其原 OR 活跃规则不能保证 full 自身用了新区域：obs300/1012/120 都是 full=0；它们只能说明回退与比较头修正的差异，不能作为 full 区域机制识别成功/失败的例证。

## 总体边界

以下直接读取既有 pooled 表，单位 pp；不重算 bootstrap，不将三seed当独立数据。

|阶段及比较（full减对照）|seed17 OA/BA|seed29 OA/BA|seed43 OA/BA|
|---|---:|---:|---:|
|源验证 full-body|+0.328 / +0.843|+0.437 / +0.986|+0.109 / -0.204|
|源验证 full-shuffled|-0.437 / +0.478|-0.328 / +0.289|+0.109 / +0.500|
|测试 full-body|-0.464 / +0.389|-0.148 / +0.426|+0.422 / +0.052|
|测试 full-shuffled|-0.063 / +0.720|-0.190 / +0.760|-0.105 / +0.013|

full 测试 BA 对这两个对照均三seed正，但 OA 对 body 两负一正，对 shuffled 三负。shuffled seed43 的三个fold全部0步，此时比较实际包含 full 对 B 的差异，不能叫该seed学到了无序区域组织。每臂九头中零步数为 full5、shuffled5、body2、body_end3、body_side4。

body_end/body_side 的源端收益不保证目标保留：源验证 body_side 相对 body 三seed OA/BA均更高，但测试 OA三seed均低于body，BA仅seed29较高；body_end 源验证相对body BA三seed均高，但测试OA三seed均低，BA有正有负。部分区域组合迁移退化是真实结果，不据此追加调参。

各头名义均853参数；保存的非恒定输入列 body41–42、body_end/body_side53–54、shuffled65–66，不能宣称有效信息/自由度相同。body只清新增端/侧摘要，B仍整图，quality仍包含端/侧覆盖；shuffled的有序quality也保留可推断身份线索。

## 原规则与补充

原规则：seed17 full-body 各结局优先至少一头非零、组内最小 obs_id；full-shuffled 首rescue/harm不额外追挑。原body全组 rescue95、harm117、CC2238、WW2294；OR活跃对应95、117、1490、1491。WW全组首ID108不活跃，按原规则选120。shuffled首rescue/harm与body重复，合并为300/1012。

追加规则在上述结果已知后由主流程明确要求：只在full-body双方epoch>0的rescue53例、harm66例中，各取最小obs_id，得到966/1598。原结果不替换。最多两图、不按分差幅度选、不修改测试或checkpoint。病例不是总体比例、不是新的盲测或独立样本证明。

`case_selection.csv` 的 `fallback` 是“所选比较是否双方都为0”，不是 full 自身回退标记；full 是否回退应读 `dynamic_epoch==0`。`phase` 区分 original 与 both_active_addendum。

## 六个实际对象

分差均固定真类减表中rival，另在 cases.csv 和图中保留自身top1 margin。比较结局是 full 相对该对照，不能不带参照地叫新增救回/误伤。

|obs / 角色|真类 / rival|B / full / body 分差|full/body步数|准确解释|
|---|---|---:|---:|---|
|300 原rescue|general2 / bulk0|+0.006932 / +0.006932 / -0.257869|0/50|full=B，避免body把已对样本改错；不是full区域救回|
|1012 原harm|chemical3 / bulk0|-0.007297 / -0.007297 / +0.269913|0/50|body救回B，full回退未救回；不是区域造成B新错|
|35 原CC|general2 / chemical3|+0.153525 / +0.455705 / +0.639752|80/95|两头均正确且都扩大B分差，body更大|
|120 原WW|general2 / chemical3|-1.250495 / -1.250495 / -1.503190|0/50|full回退，body加重既有错误；未恢复类别证据|
|966 补rescue|container4 / chemical3|-0.072063 / +0.162812 / -0.245252|80/95|full确实救回B和body都错的样本，非零头实例|
|1598 补harm|bulk0 / container4|-0.278019 / -0.281196 / +0.620199|80/95|body救回B，full非零但仍错，略加重原分差；不是full新误伤B|

300另一比较的shuffled实际误判为fishing1，因此独立使用rival1：B/full `+0.148094`，shuffled `-0.309378`，不能沿用body错误类0。1012的shuffled也预测正确3，rival0分差 `+0.069088`。两者shuffled都是60步。图上主摘要柱固定为首比较的rival，文字区分别列两个实际比较rival，不混用。类别名已与cohort.csv核对，原油轮为class5而非class1。

### 实际原图观察

- **300，Rotterdam，VV/VH均64×64**：两极化均见窄小斜向亮目标，VV局部亮点更集中，VH主体较连续。body/end/side均有合法支持。full虽然缓存有非零端/侧摘要，却因零头完全不用它们；不能由摘要柱正负编造其保护机制。
- **1012，Jebel Ali，VV72×72、VH64×64**：VV中主体很亮且有明显横纵延伸，VH亮主体之外对比不同；各极化区域覆盖使用各自原窗而非强行配准。full零步，body和shuffled都救回，图像的外围可见差异不等于外围摘要是救回必要条件。
- **35，Tanger Med，VV/VH64×64**：紧凑亮主体置于明显纹理背景，三个区域均有效。VV/VH同一统计的真类减竞争类符号并不一致，两个非零头最终都保留真类2。这里rival是chemical3，不能沿用此前full-fixed病例的实际错类7。
- **120，Tanger Med，VV/VH64×64**：斜向细长亮主体，VV背景纹理比VH明显，三区均有效。虽然端/侧摘要非零且部分统计方向相反，full为0步，持续错误不能归因full训练后的区域读出。
- **966，Port Klang，VV/VH64×64**：两极化均见细长亮主体，VV主体外可见轴向亮延伸，VH外部对比更弱。真container4相对chemical3，body原始mu为负而h为正；新增端/侧h也有正响应，其他统计并非全同向。full80步将分差越过0，body95步未救回，提供三区摘要联合训练存在可用个例的证据。柱图是未标准化输入而非贡献，不能把其中某根柱直接当救回原因。
- **1598，Santos，VV/VH64×64**：两极化均是长条强亮主体，周围有纹理且不存在“缺图导致失败”的直接解释。真bulk0相对container4，VV body mu很正、h负；VH多个统计为正，但full仍预测container。body95步反而救回，直接说明原始方向响应正负和最终非线性决策不能等同，也不能宣称保留全部区域必然比body好。

六个对象两极化及三区均有效。本轮仅观察图像可见结构与模型输入/输出，不命名物理散射原因；layer1 RF35会跨区域，小区域摘要不等于孤立船体/背景像素。两头独立训练且选步不同，病例不隔离单区域的独立因果贡献，本轮没有额外干预来制造这种解释。

## 交付与停止点

`review_cases.py` 复用旧 `one_info`、native/crop-pad/soft_regions 和已缓存摘要；`cases.csv` 保存分差/图像原路径/epoch，`case_slots.csv` 保存实际输入各类mu/h/b及shuffled槽来源，`replay_checks.csv` 保存独立输入比较与分数误差，`case_selection.csv` 保留组数和原/补充规则。`case_300/1012/35/120/966/1598.png` 共六张均已实际查看。

本轮脚本最后一次完整运行约8.28秒，仅缓存载入、CPU小头重放和绘图，不是新增推理成本或端到端测速。无本任务运行进程。到此收口，不新增模型、病例、bootstrap或参数搜索。
