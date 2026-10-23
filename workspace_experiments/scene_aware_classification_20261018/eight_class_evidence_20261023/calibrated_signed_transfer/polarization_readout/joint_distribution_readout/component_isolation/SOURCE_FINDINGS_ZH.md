# 源端组成分离：18头已冻结

2026-10-23，以下内容在追加test推理前写定。18头，3600步，训练29.06秒、运行36.78秒，均为CPU缓存小头范围；没有重新提取特征、reader或校准器拟合，不代表完整SAR系统延迟。

## 主要结果

复用原additive的Head/inputs，名义266参数；另一个统计及其跨类中心化槽在标准化后置0。原full9、mean_only9直接复用。原B及full源验证重放最大误差均为0，源scaler与原full相等，两新臂同seed初始化逐张量相同。只有信息遮槽不同，原训练/源J规则不变。

每seed源验证915条fold-instance，816唯一对象、629 MMSI，不是915独立船。三个fold验证310/286/319，训练4392/4419/4418；每fold八类齐全，源验证均为2025-09，训练早于2025-09。valid VH验证274/265/294。各fold源train/val身份无交叠；原矩inner OOF及旧AD OOF身份排除沿用并检查。冻结T仍非全栈OOF。

|臂|seed17 OA/BA|seed29 OA/BA|seed43 OA/BA|相对B净变化17/29/43|
|---|---|---|---|---|
|B|54.863/46.830|57.158/47.863|53.989/46.012|0/0/0|
|原full|56.284/48.423|57.923/48.907|55.519/47.213|+13/+7/+14|
|sigma_only|55.738/47.619|57.486/48.746|55.410/47.142|+8/+3/+13|
|kappa_only|56.721/48.280|58.033/48.782|56.066/46.890|+17/+8/+19|
|原mean_only|57.814/48.171|58.361/48.214|55.956/46.838|+27/+11/+18|

full对sigma_only的OA/BA三seed均高，相对多判对5/4/1。kappa_only对full多判对4/1/5，但BA低.143/.125/.324pp。full对mean_only也是OA较低、BA三seed较高，不能称全面胜出。

full对sigma的NLL/Brier三seed均更低；kappa对full的NLL差为+.018426/+.007508/-.009019，Brier差+.006503/+.002613/-.005132。概率指标未与OA统一排序，详见source_pooled及source_paired。

## 类别与选点

相对full，sigma在bulk净-4/-5/-9，chemical +2/0/+6；kappa在bulk -3/-1/-2，chemical +9/+3/+9。其它类有小幅正负，完整逐类及fold/pass/port/valid结果保存在source_stratified及source_paired_groups。故目前不是一个统计对所有类更优，而是源目标与类频率下的取舍。

sigma选步按fold排列：45/15/150、60/65/5、0/5/15；kappa：60/35/165、10/15/0、95/5/125，每组三数为17/29/43。两头为epoch0，分别sigma fold2 seed17、kappa fold1 seed43。训练曲线和所选源train预测已保存；train指标是这些监督头在训练输入上的拟合表现，不是新的OOF误差估计。

## 实际源VV/VH回看

source_panels的8幅已全部用view_image查看，只有5个唯一对象。均按seed17、instance_id再fold首例，没有追挑。

- sigma救回1668：64px，两极化均可见斜向较长的饱和亮条；bulk对offshore固定竞争分差从-.50818变+.03260。mean也救回、full未救回，不能将此例视为sigma的独有信息证明。
- kappa救回0415：64px，横向亮条，VV框外另有亮点；bulk对chemical分差从-.02357变+.06686。mean也判对，full为零增量；看见框外亮点不等于它造成救回。
- 两臂误伤0474：64px，中央短亮团，VV背景比VH更亮；general对tug原分差仅+.01080，sigma变-.04042、kappa变-.04799。full也错而mean保留正确，表明同样输入形态下小分差调整有代价。
- 两臂CC为pair_0032734：72px，近竖直亮条，B已明显正确。sigma epoch0保持+2.08997分差，kappa训练后变+2.17611；前者不是学习保护证据。
- 两臂WW为pair_0032903：64px，较暗的狭长响应，VH较连贯；bulk仍错为general。sigma epoch0不变，kappa将分差从-1.39142改善至-1.27578但未翻转。不凭暗亮差异命名物理机制。

图题包含实际竞争类、full/mean预测和新臂选步；source_case_slots保存同outer reader两极化mu/sigma/kappa供追溯。以上是代表例，不当总体比例。

## 源端能回答与不能回答

在该固定预算下，单独sigma尚未替代完整full的源OA/BA；保留kappa但删sigma会改变类别取舍，不能由其较高OA宣称sigma无用。kappa包含幅度而非纯符号，屏蔽实验改变信息输入及学习路径，不能单凭这18头证明物理正负机制或数学必要性。

能量和/差恒等式在每pol每旋转内成立；缓存平均统计不是平均能量的可逆表述。两臂名义容量相同但有效可变信息较少；mean_only架构209参数，也不是严格等容量臂。此源验证已经反复开发，本轮不作确认性CI。追加test授权及其非盲测边界另见TEST_ADDENDUM_ZH.md，不能回填或改写这里的源结论。
