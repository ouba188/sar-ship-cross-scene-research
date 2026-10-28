# 本轮数据摘要

同一layer2内的受控开发位置比较：136条源适配、160条历史开发查询、16条固定参考。开发集反复使用，不声称独立源验证选择，更不是未见目标盲测；3075扩围不参加本轮。三seed是匹配训练随机条件，不是三份独立数据；不平均概率作为主结果。OA/BA以百分数表示，差值为百分点；BA只平均本开发集实际出现的7类。NLL为自然对数交叉熵，Brier为8类概率平方误差之和，二者越低越好。

| split | 观测 | 船只 | 过境 | 港口 | 双极化 | 仅VV |
|---|---:|---:|---:|---:|---:|---:|
| source_adapt | 136 | 128 | 20 | 10 | 128 | 8 |
| development | 160 | 158 | 3 | 3 | 145 | 15 |
| references | 16 | 16 | 13 | 9 | 15 | 1 |

| 类别 | source_adapt | development | references |
|---|---:|---:|---:|
| bulk_carrier | 54 | 25 | 2 |
| fishing | 4 | 0 | 2 |
| general_cargo | 26 | 12 | 2 |
| chemical_tanker | 28 | 70 | 2 |
| container | 6 | 22 | 2 |
| crude_oil_tanker | 12 | 16 | 2 |
| tug | 3 | 7 | 2 |
| offshore | 3 | 8 | 2 |

开发港口：Jebel Ali=69；Port Klang=28；Singapore=63。

| split pair | 重叠ID | 重叠船只 | 重叠过境 | 重叠港口 |
|---|---:|---:|---:|---:|
| source_adapt__development | 0 | 0 | 0 | 0 |
| source_adapt__references | 0 | 0 | 0 | 2 |
| development__references | 0 | 0 | 0 | 3 |

这些是不交叉/覆盖的实际统计，不改变历史开发集被反复使用的事实。
reference_eligibility.csv记录每查询经ID/MMSI排除后的实际参考及同类参考。
data_groups.csv保留分割内类别、港口、过境及manifest极化分组；极化分组不等于单独VV/VH推理消融。
data_preview内为4个固定输入例的VV/VH及16条实际参考库；原始8bit显示chip与模型160裁剪/补零并列，1:1像素、不伸缩、不拉伸对比度，不是原始SAR辐射值。
native_sources.json记录路径、有效性及裁剪来源。图库已生成但本支线未人工查看；可见事实与路径解释待主代理补充。
