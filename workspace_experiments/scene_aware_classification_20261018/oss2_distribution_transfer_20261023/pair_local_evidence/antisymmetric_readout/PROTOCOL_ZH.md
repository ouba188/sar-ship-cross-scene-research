# 源端八类反对称读出接口检验

2026-10-23。先锁协议再运行。只用fold0 source train1542/validation279，复用source_moments.npz的全部28类对和8类边缘统计；不提取新图像、不使用test进行推理、评价或选择。当前三对logistic弱信号不当作完整八类效用证明。

## 固定接口

每个无序类对c<d取15槽：

`[B_c-B_d, muVV_c-muVV_d, muVH_c-muVH_d, sigmaVV_c, sigmaVV_d, kappaVV_c, kappaVV_d, sigmaVH_c, sigmaVH_d, kappaVH_c, kappaVH_d, sigmaPairVV, kappaPairVV, sigmaPairVH, kappaPairVH]`。

反向交换边缘统计c/d槽；前三差值与pair kappa取负，pair sigma不变。正反向训练tokens共同拟合逐槽均值/标准差，最低尺度1e-8，验证复用，不用标签选竞争者。复用原始矩，不沿用上一轮asinh修正。

共享15→12 GELU→1，205参数，末层零初始化。h_cd=.5(f(x_cd)-f(x_dc))，residual_c+=h_cd/7、residual_d-=h_cd/7。固定八类全部28对，没有gate、温度、lambda或GT选对。输出是B+residual。

三臂相同结构/初始化：global仅保留前三槽，其余原始槽置0；marginal保留前11槽，pair槽置0；paired保留15槽。名义容量均205，但有效信息和活跃输入维数不同，如实报告。消融先于scaler，禁用槽在正反向和验证均为0。

每seed17/29/43分别从calibrated/fold0_seedXX取得T(A OOF)源训练B及outer源验证B，绝不把source_moments内seed17的base用于其他seed。A/T/W和缓存固定，不修改父脚本globals。

## 训练、选择与边界

9head，200步jointCE=.5普通CE+.5八类均衡CE，AdamW lr .003/wd .01，每5步仅源验证精确J=(OA+BA)/2选点，含0、严格更优才更新，平局最早。没有额外参数搜索。所有训练完成后给出三臂train/validation配对OA/BA/NLL及其他原指标。不自动扩到test，也不以验证最好者触发后续训练。

训练输入的辅助读出是inner OOF，但校准T和本头都使用这些源标签拟合，不能把头的train表现叫整流水线OOF。validation参与选点，只是源开发结果，不是独立外部验证。

## 验证与交付

核对sample_ids到全cohort的映射，source train/validation精确匹配reader索引；训练held reader及MMSI/pass隔离、验证outer映射正确。检查实际正反向raw输入、h翻号、类顺序置换等变、残差和0、零初始化回退B；使用selected权重和真实源输入执行数值检查，不只写符号声明。checkpoint重放与选中validation J一致。

保存heads、scaler、pairs、源索引/source_rows、predictions、selections、curves、metrics、paired_metrics、结构检查和sourcecases接口。sourcecases仅提供seed17源验证各臂四结局首ID供已指定分析代理回看，不渲染或重新挑选案例。该接口属于相对比较后合成的检验，不宣称新卷积、首创协方差或已证实的物理机制。

按训练前补充要求，每个selected头保存279条validation的28个h_cd、B和最终s至validation_flux。validation_paths同时列旧/新最强错误类，并对新错误类r精确分解真类分差增量：2h_yr/7与其余六类边的sum_k(h_yk-h_rk)/7。不能把完整救回都归于直接y-r边。此项已随原9头执行，不追加训练。
