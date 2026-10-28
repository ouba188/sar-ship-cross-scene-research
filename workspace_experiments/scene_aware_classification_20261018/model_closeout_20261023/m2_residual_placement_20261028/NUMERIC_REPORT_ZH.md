# M2 C/D/E位置实验：CPU数值报告

136条源适配样本的小范围研发；160条反复复用的开发集不是独立确认集。三seed逐一匹配baseline，不将模型平均概率作为主结果；seed不是独立船只或独立数据集。OA/BA为百分数，差值为百分点；BA仅平均开发集中实际出现的7类。NLL为自然对数交叉熵，Brier为8类概率平方误差之和；后二者越低越好。参考库固定16条、每类2条，其中1条无VH；不据外观命名物理原因。

C=preadd：conv3+BN后、identity相加前；D=postadd：相加后、ReLU前；E=stage：ReLU后。
C/D/E均512通道、每臂新增197328参数；仅位置与其输入/传播路径不同。
协议为6次新训练（C/D各3seed）、6个复用模型（baseline/E各3seed）；实际运行记录见summary.json的runs。

## 每臂均值

| arm | OA | BA | NLL | Brier |
|---|---:|---:|---:|---:|
| baseline | 45.000000 | 34.264842 | 2.532967 | 0.843748 |
| stage | 46.458333 | 36.972635 | 2.530360 | 0.835287 |
| preadd | 46.666667 | 36.054267 | 2.517617 | 0.829558 |
| postadd | 46.250000 | 35.918213 | 2.517572 | 0.829591 |

## 当前判断

preadd相对E：OA +0.208333、BA -0.918367个百分点；NLL -0.012742、Brier -0.005729。三seed中有利方向个数依次为1/0/3/3。

postadd相对E：OA -0.208333、BA -1.054422个百分点；NLL -0.012787、Brier -0.005696。三seed中有利方向个数依次为0/0/3/3。

C/D呈现更好的概率质量与较低BA的取舍，不能据此全面替代E；等容量比较支持位置影响类别竞争，但不单独证明ReLU或物理成因。


## 逐seed结果

| arm | seed | OA | BA | NLL | Brier | dOA | dBA | dNLL | dBrier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 17 | 44.375000 | 33.831169 | 2.541080 | 0.840238 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| baseline | 29 | 45.625000 | 34.928108 | 2.517699 | 0.845445 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| baseline | 43 | 45.000000 | 34.035250 | 2.540120 | 0.845561 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| stage | 17 | 46.250000 | 36.659709 | 2.525600 | 0.830681 | 1.875000 | 2.828541 | -0.015480 | -0.009557 |
| stage | 29 | 46.250000 | 37.027056 | 2.520337 | 0.838642 | 0.625000 | 2.098949 | 0.002638 | -0.006803 |
| stage | 43 | 46.875000 | 37.231138 | 2.545142 | 0.836540 | 1.875000 | 3.195887 | 0.005022 | -0.009022 |
| preadd | 17 | 46.250000 | 35.673315 | 2.516895 | 0.826156 | 1.875000 | 1.842146 | -0.024185 | -0.014081 |
| preadd | 29 | 46.250000 | 36.040662 | 2.501292 | 0.829944 | 0.625000 | 1.112554 | -0.016406 | -0.015501 |
| preadd | 43 | 47.500000 | 36.448825 | 2.534665 | 0.832575 | 2.500000 | 2.413575 | -0.005456 | -0.012986 |
| postadd | 17 | 46.250000 | 35.673315 | 2.517139 | 0.825898 | 1.875000 | 1.842146 | -0.023942 | -0.014340 |
| postadd | 29 | 46.250000 | 36.040662 | 2.499475 | 0.829751 | 0.625000 | 1.112554 | -0.018223 | -0.015694 |
| postadd | 43 | 46.250000 | 36.040662 | 2.536104 | 0.833125 | 1.250000 | 2.005411 | -0.004017 | -0.012436 |

## 同seed配对

逐seed计算差值后取均值和样本标准差；不平均概率、不把480次预测当成480条独立样本。

| comparison | dOA | dBA | dNLL | dBrier | favorable seeds OA/BA/NLL/Brier |
|---|---:|---:|---:|---:|---|
| stage__vs__baseline | 1.458333 +/- 0.721688 | 2.707792 +/- 0.558349 | -0.002607 +/- 0.011212 | -0.008461 +/- 0.001460 | 3/3/1/3 |
| preadd__vs__baseline | 1.666667 +/- 0.954703 | 1.789425 +/- 0.652111 | -0.015349 +/- 0.009409 | -0.014190 +/- 0.001261 | 3/3/3/3 |
| postadd__vs__baseline | 1.250000 +/- 0.625000 | 1.653370 +/- 0.475422 | -0.015394 +/- 0.010259 | -0.014157 +/- 0.001636 | 3/3/3/3 |
| preadd__vs__stage | 0.208333 +/- 0.360844 | -0.918367 +/- 0.117827 | -0.012742 +/- 0.005529 | -0.005729 +/- 0.002587 | 1/0/3/3 |
| postadd__vs__stage | -0.208333 +/- 0.360844 | -1.054422 +/- 0.117827 | -0.012787 +/- 0.006999 | -0.005696 +/- 0.002850 | 0/0/3/3 |
| postadd__vs__preadd | -0.416667 +/- 0.721688 | -0.136054 +/- 0.235653 | -0.000045 +/- 0.001647 | 0.000033 +/- 0.000449 | 0/0/1/2 |

## 救回与误伤

事件数跨3seed累加；独立ID/船只数另列，同一ID可以在不同seed属于两组。

| comparison | rescue events/IDs/ships | harm events/IDs/ships | both IDs |
|---|---|---|---|---|
| stage__vs__baseline | 17/8/8 | 10/4/4 | 0 |
| preadd__vs__baseline | 14/7/7 | 6/3/3 | 0 |
| postadd__vs__baseline | 13/7/7 | 7/3/3 | 0 |
| preadd__vs__stage | 7/4/4 | 6/3/3 | 0 |
| postadd__vs__stage | 5/3/3 | 6/3/3 | 0 |
| postadd__vs__preadd | 0/0/0 | 2/2/2 | 0 |

## 逐类召回变化

以下为三seed平均百分点差；缺失类保留NA。

| comparison | class | n | dRecall | dNLL | dBrier |
|---|---|---:|---:|---:|---:|
| stage vs baseline | bulk_carrier | 25 | 2.666667 | -0.031047 | -0.014743 |
| stage vs baseline | fishing | 0 | NA | NA | NA |
| stage vs baseline | general_cargo | 12 | 16.666667 | -0.253522 | -0.094988 |
| stage vs baseline | chemical_tanker | 70 | 0.000000 | 0.050404 | 0.018598 |
| stage vs baseline | container | 22 | -4.545455 | 0.025839 | -0.016049 |
| stage vs baseline | crude_oil_tanker | 16 | 4.166667 | -0.221453 | -0.060719 |
| stage vs baseline | tug | 7 | 0.000000 | 0.020219 | 0.001457 |
| stage vs baseline | offshore | 8 | 0.000000 | 0.338290 | 0.020909 |
| preadd vs baseline | bulk_carrier | 25 | 2.666667 | -0.039512 | -0.007949 |
| preadd vs baseline | fishing | 0 | NA | NA | NA |
| preadd vs baseline | general_cargo | 12 | 8.333333 | -0.199403 | -0.051439 |
| preadd vs baseline | chemical_tanker | 70 | 1.904762 | -0.018221 | -0.014583 |
| preadd vs baseline | container | 22 | -4.545455 | 0.065873 | -0.005785 |
| preadd vs baseline | crude_oil_tanker | 16 | 4.166667 | -0.162141 | -0.034046 |
| preadd vs baseline | tug | 7 | 0.000000 | 0.071746 | 0.003358 |
| preadd vs baseline | offshore | 8 | 0.000000 | 0.355389 | 0.026870 |
| postadd vs baseline | bulk_carrier | 25 | 2.666667 | -0.042262 | -0.012505 |
| postadd vs baseline | fishing | 0 | NA | NA | NA |
| postadd vs baseline | general_cargo | 12 | 8.333333 | -0.243008 | -0.064967 |
| postadd vs baseline | chemical_tanker | 70 | 0.952381 | -0.005169 | -0.009979 |
| postadd vs baseline | container | 22 | -4.545455 | 0.078412 | -0.006210 |
| postadd vs baseline | crude_oil_tanker | 16 | 4.166667 | -0.178126 | -0.029207 |
| postadd vs baseline | tug | 7 | 0.000000 | 0.074338 | 0.002389 |
| postadd vs baseline | offshore | 8 | 0.000000 | 0.309504 | 0.014111 |
| preadd vs stage | bulk_carrier | 25 | 0.000000 | -0.008465 | 0.006795 |
| preadd vs stage | fishing | 0 | NA | NA | NA |
| preadd vs stage | general_cargo | 12 | -8.333333 | 0.054118 | 0.043549 |
| preadd vs stage | chemical_tanker | 70 | 1.904762 | -0.068625 | -0.033181 |
| preadd vs stage | container | 22 | 0.000000 | 0.040035 | 0.010264 |
| preadd vs stage | crude_oil_tanker | 16 | 0.000000 | 0.059312 | 0.026673 |
| preadd vs stage | tug | 7 | 0.000000 | 0.051526 | 0.001901 |
| preadd vs stage | offshore | 8 | 0.000000 | 0.017099 | 0.005960 |
| postadd vs stage | bulk_carrier | 25 | 0.000000 | -0.011215 | 0.002238 |
| postadd vs stage | fishing | 0 | NA | NA | NA |
| postadd vs stage | general_cargo | 12 | -8.333333 | 0.010514 | 0.030020 |
| postadd vs stage | chemical_tanker | 70 | 0.952381 | -0.055573 | -0.028577 |
| postadd vs stage | container | 22 | 0.000000 | 0.052573 | 0.009839 |
| postadd vs stage | crude_oil_tanker | 16 | 0.000000 | 0.043327 | 0.031512 |
| postadd vs stage | tug | 7 | 0.000000 | 0.054118 | 0.000933 |
| postadd vs stage | offshore | 8 | 0.000000 | -0.028786 | -0.006798 |
| postadd vs preadd | bulk_carrier | 25 | 0.000000 | -0.002750 | -0.004556 |
| postadd vs preadd | fishing | 0 | NA | NA | NA |
| postadd vs preadd | general_cargo | 12 | 0.000000 | -0.043604 | -0.013528 |
| postadd vs preadd | chemical_tanker | 70 | -0.952381 | 0.013052 | 0.004604 |
| postadd vs preadd | container | 22 | 0.000000 | 0.012538 | -0.000425 |
| postadd vs preadd | crude_oil_tanker | 16 | 0.000000 | -0.015984 | 0.004839 |
| postadd vs preadd | tug | 7 | 0.000000 | 0.002592 | -0.000969 |
| postadd vs preadd | offshore | 8 | 0.000000 | -0.045885 | -0.012758 |

## 案例与边界

seed17按六组比较分别选择持续判对、持续判错、救回、误伤；空组如实保留0，不从其他seed补齐。
每组按初始模型固定竞争类分差绝对变化最大选1例，ID字典序打破平局；包含D对C、C/D对E的直接比较。另补主线path_diagnosis选例，以及其他seed实际存在的D/C翻转代表例；补充例不计入seed17分组。
cases.json同时保存全部3seed预测和固定竞争类分差、实际同类/竞争类参考ID；native_sources.json记录路径及有效性。
path_diagnosis的竞争类按baseline seed17定义；图库主分差按initial定义，两者分开记录，并补齐诊断竞争类参考。
[原始VV/VH及实际参考图库](CASES/index.html)。生成图像不代表已人工看图，本CPU报告不作图像物理成因判断。
paired_predictions.csv保留所有逐样本配对；classes.csv、grouped.csv含类别及港口/过境/极化分层。
bypass.csv为同权重关闭新增回写的诊断，不能替代重新训练对照；主线负责路径干预及最终解释。
仅3个开发过境且开发集反复使用，结果不等于独立跨场景确认；不以训练秒数推断端到端效率。
