# 已选头案例与摘要干预

本规则在制图前写入，沿用主流程在test成绩发布前指定的active机制展示规则。不训练、不改epoch、不按增益幅度选样本。范围为seed17历史test四个结局，比较dynamic与fixed。

- 救回/误伤优先两头epoch>0的对应组，按obs_id、instance_id、fold升序首例。
- CC/WW优先dynamic epoch>0的对应组，同样取首例。
- 若active组为空，则使用该结局全组首例并注明零步回退；全组不存在则明确空缺，不借source/test或其他epoch补例。
- 并列记录全组首ID与本规则首ID及组数量，不把active子集当总体结果。
- 救回固定竞争类为fixed实际错误类；误伤为dynamic实际错误类。WW用dynamic实际错误类；CC用dynamic最高非真类。另存两模型各自top1竞争分差，所有干预保持固定竞争类不变。

恢复真实checkpoint及scaler，在缓存outer摘要上先重放完整分数。对dynamic及fixed分别做3次区域摘要干预：对应body/end/side的两个pol、8类、mu/h/b全部置原始数值0，再调用原raw_inputs重新构造跨类中心化槽；B、其他区域、geometry/quality/region_valid、scaler及head保持固定。原始摘要0不同于标准化槽0；不把它称区域物理删除或重新训练消融。非线性下3个删除效果不要求可加，残留quality与head偏置意味着摘要置零未必回到B。若原头epoch0，完整与干预分数均必须严格等于B。

每例复用原PNG、原crop_pad、实际outer固定模板和真实OBB，显示原VV/VH与两种区域。只读取缓存和原图，不重提骨干、不做训练、标签更改或新网格。结果全部保留。

## 主流程指定的两例补充

主流程实际查看上述四图后指出它们没有覆盖相对s_keep的新救回/新误伤，随后明确追加obs966（真4，B错3→dynamic对4）及obs2217（真3，B对3→dynamic错6）。二者均为主analysis/test_cases.csv相对baseline对应组的最小obs_id，不按分差大小挑选。固定rival分别为3、6，使用同样checkpoint/scaler/区域摘要置零规则，不新增训练或更换epoch。原四例仍标相对fixed，新增两例单独标相对baseline，总6例。新增运行仅计算两例，保留原四例结果。
