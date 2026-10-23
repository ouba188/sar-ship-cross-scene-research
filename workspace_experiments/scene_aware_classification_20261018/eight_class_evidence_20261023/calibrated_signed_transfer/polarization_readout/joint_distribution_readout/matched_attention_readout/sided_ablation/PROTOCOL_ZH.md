# 单侧必要性：冻结协议

2026-10-23，新增fit之前。当前授权仅high/low各9头，共18fit；原two_sided_attention9头只复用。只写本sided_ablation目录，不改父文件/Git，不增加τ/尺度/宽度/epoch搜索。

## 信息干预

读取父源/test合法角色attention缓存：h=(q_plus-q_minus)/2，b=(q_plus+q_minus)/2。因此逐C4平均缓存可精确线性恢复平均q_plus=h+b、q_minus=b-h，无需重提图。

- high分布两原始槽固定为[q_plus,0]。
- low分布两原始槽固定为[0,q_minus]。
- 从上述新槽重新生成类中心化槽，再仅用各fold源OOF训练tokens拟合scaler。零槽及其中心槽严格为0，不复用旧h/b scaler，不将另一侧通过h/b或其中心化值泄漏。
- 共同mean、center(B)、valid不变；同B/W/τ、原inner-held/outer及MMSI/采集规则。source/test文件索引合法角色。σ_e/κ_e不是本轮训练输入。

每臂复用父additive结构266名义参数、同seed17/29/43初始张量、零末层；一侧槽置零令有效输入/可学习自由度少于双侧，因此这是受限信息学习对照，不是有效容量严格相同或纯因果删除。high/low两臂彼此有效维数相同。两侧h/b与q_plus/q_minus是可逆线性坐标，但单侧保留一个投影；不把初始化下坐标条件差别排除为零。

## 学习及阶段顺序

3fold×3seed×2arms，200steps全批次，jointCE=.5ordinaryCE+.5balancedCE，AdamW lr=.003/wd=.01。source J=(OA+BA)/2每5步检查，含0，严格大于才更新，平局最早。不得用test选点、温度或臂。

先完成全部源train/validation预测、选点/曲线/源报告并保存冻结时间；随后固定历史test4744一次推理，不新增fit。复用原two-sided预测而非重训；测试开始后不改变任何头。

## 输出与必要检查

保存18checkpoint、源/目标预测、OA/BA/NLL/Brier、逐类/seed/采集以及双侧减high/low和high减low配对。不新增bootstrap网格。检查源OOF覆盖、computed角色、q恢复、零槽及scaler、共同输入、同seed初始化、0步=B、保存重放J。训练计时明确为缓存适配wall time而非纯优化器时间。

原图固定test四ID5030/7310/3494/3504并实际view各自VV/VH，显示B/双侧/high/low真实输出及单侧输入。若四例未反映新增侧间差异，再只加seed17按ID/fold首个high对low错和low对high错各1；这是high相对low的独有救回/误伤，不混称相对B。无该结局则记录，不挑替代漂亮图。

原T非二级OOF、源验证及历史test参与过开发，非独立外部确认。q_plus/q_minus是学习类别响应两侧，不是物理正确/错误散射。完成正负结果即收口。
