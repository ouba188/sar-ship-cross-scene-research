"""Summarize frozen source-trained ring-use probes without selecting target winners."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from probe_ring_predictability import OUT, aggregate, risk_weights


def markdown(frame, decimals=4):
    columns=list(frame.columns)
    lines=["| "+" | ".join(columns)+" |", "| "+" | ".join(["---"]*len(columns))+" |"]
    for row in frame.itertuples(index=False,name=None):
        values=[]
        for value in row:
            if isinstance(value,(float,np.floating)):
                values.append("NA" if not np.isfinite(value) else f"{value:.{decimals}f}")
            else:
                values.append(str(value))
        lines.append("| "+" | ".join(values)+" |")
    return "\n".join(lines)


def main():
    source=pd.read_csv(OUT/"source_scored.csv",dtype={"mmsi":str})
    target=pd.read_csv(OUT/"target_scored.csv",dtype={"mmsi":str})
    events=pd.read_csv(OUT/"event_diagnostics.csv")
    units=pd.read_csv(OUT/"target_metrics.csv")
    ports=pd.read_csv(OUT/"target_per_port.csv")
    summary=pd.read_csv(OUT/"target_summary.csv")
    choices=json.loads((OUT/"SOURCE_FROZEN.json").read_text(encoding="utf-8"))
    verify=json.loads((OUT/"DONE.json").read_text(encoding="utf-8"))
    source_metric=pd.read_csv(OUT/"source_metrics.csv")
    source_summary=aggregate(source_metric)

    weighted=[]
    for frame in (source,target):
        for keys,g in frame.groupby(["pool","split","budget","origin"]):
            w=risk_weights(g)
            mask=g.gain.ne(0).to_numpy()
            y=g.loc[mask,"gain"].eq(1)
            for model in ("logistic","tree"):
                weighted.append(dict(zip(["pool","split","budget","origin"],keys),model=model,
                    auc_ba_weighted=roc_auc_score(y,g.loc[mask,"score_"+model],sample_weight=w[mask]) if y.nunique()==2 else np.nan))
    events=events.merge(pd.DataFrame(weighted),on=["pool","split","budget","origin","model"],validate="one_to_one")
    events.to_csv(OUT/"event_diagnostics_weighted.csv",index=False)

    delta=summary.merge(summary.loc[summary.arm.eq("body")].drop(columns="arm"),
        on=["budget","origin"],suffixes=("","_body"))
    for metric in ("oa","ba","macro_nll","macro_brier"):
        delta["delta_"+metric]=delta[metric]-delta[metric+"_body"]
    delta.to_csv(OUT/"target_effects_vs_body.csv",index=False)
    pdeltas=ports.pivot(index=["pool","budget","origin"],columns="arm",values="ba")
    pdeltas=pdeltas.subtract(pdeltas.body,axis=0).reset_index()
    pdeltas.to_csv(OUT/"per_port_ba_effects.csv",index=False)
    draws=aggregate(units,["draw"])
    draws=draws.pivot(index=["pool","budget","origin","draw"],columns="arm",values="ba")
    draws=draws.subtract(draws.body,axis=0).reset_index()
    draws.to_csv(OUT/"per_port_draw_ba_effects.csv",index=False)

    changed=target.loc[target.origin.eq("cross")&target.gain.ne(0)].copy()
    changed["gain_selected"]=changed.gain*changed.choose_logistic
    changed.groupby(["pool","budget","truth"]).agg(rescues=("gain",lambda x:int((x==1).sum())),
        harms=("gain",lambda x:int((x==-1).sum())),selected_net=("gain_selected","sum"),
        ships=("mmsi","nunique")).to_csv(OUT/"class_event_counts.csv")
    # Target cases selected for explanation, never for altering the frozen policy.
    sg=changed.loc[changed.pool.eq("Singapore")&changed.budget.eq(64)]
    sg.to_csv(OUT/"singapore_B64_events.csv",index=False)
    event_source=events.loc[events.origin.eq("cross")&events.model.eq("logistic")&events.split.ne("target")]
    compact=event_source[["pool","split","rescue","harm","event_ships","event_acquisitions","auc","auc_ba_weighted"]]
    result=delta.loc[delta.origin.eq("cross"),["budget","arm","oa","ba","delta_oa","delta_ba"]]
    target_diag=events.loc[events.origin.eq("cross")&events.model.eq("logistic")&events.split.eq("target"),
        ["pool","budget","rescue","harm","event_ships","auc","auc_ba_weighted","selected_rescue","selected_harm"]]
    selected_rows=[]
    for port,c in choices.items():
        selected_rows.append(dict(pool=port,selected=c["chosen"],**c["source_selection_delta_ba"]))
    src_effect=source_summary.pivot(index=["pool","split","origin"],columns="arm",values="ba")
    src_effect=src_effect.subtract(src_effect.body,axis=0).reset_index()
    src_effect.to_csv(OUT/"source_ba_effects.csv",index=False)
    coefficient=pd.read_csv(OUT/"logistic_coefficients.csv")
    direction=coefficient.groupby("feature").agg(positive_pools=("coefficient",lambda x:int((x>0).sum())),
        negative_pools=("coefficient",lambda x:int((x<0).sum())),mean_abs=("coefficient",lambda x:float(x.abs().mean())))
    direction.to_csv(OUT/"coefficient_directions.csv")
    text=f"""# 环带收益规律能否学到：第一轮结果

## 结论先说

**源端能拟合出区分救回/误伤的规律，但目前没有形成稳定、可部署的跨港选择规律。局部正信号保留，不把全部路线判死。**

- 固定逻辑回归的跨采集BA相对船体：B16 +0.100、B32 +0.023、B64 -0.701个百分点。B16是同预算验证；B32/B64是B16训练后的预算外迁移。
- B16同频率随机选择为+0.137，超过学习器的+0.100；不能将这一小幅涨点归功于选对了查询。
- 新加坡B64逻辑回归有+0.711 BA、较固定组合+0.159、较同频率随机+0.150的局部增量，但五次draw仅3次较船体为正。额外优于固定组合主要是避免一条散货船的一次误伤，不能描述为普遍识别出了补证条件。
- 源端自动选出的策略在B16/B32/B64为+0.048/+0.014/-1.109 BA；并没有把目标端的局部机会可靠挑出来。新加坡源验证只有0次救回、4次误伤，选了不启用，错过目标B64的局部收益。

## 实际做了什么

五个留港池各训练逻辑回归和二层树，合计10个小探针。逻辑回归固定C=0.1，树最多两层、叶节点至少10个事件；没有搜索阈值，固定0.5。骨干、源先验、支持核和完整分类读出都不变。两个端点是原有船体局部读出与船体+环带读出。

18个输入只含完整预测概率、候选预测类别对应的支持数量/相似度和局部支持关系稳定性。最后一个不是重新拟合完整分类器的leave-one-out准确率，而是每次删去一个支持后，局部top3类相似度的预测类别是否变化。查询真值只用于源端训练目标和评价。

先完成全部源端学习/选择并写入SOURCE_FROZEN.json，再计算目标特征与评价。五池不能合并成一套选择器，因为不同池之间有真实船只和采集重叠；本轮始终按池隔离。

任务仍为五港，B16/B32/B64分别20/19/10个港口-采集任务，各五次支持抽样，另有同采集对照。目标{target.sample_id.nunique()}个不同查询样本、{target.mmsi.nunique()}条不同MMSI；共{len(target)}行是多预算、支持抽样和同跨采集的重复暴露。没有扩公开数据集。

## 第一层：学到了什么

下表的事件是组合相对船体一方对、一方错；两方都错没有OA/BA选择偏好，未用于拟合，但完整任务评价始终保留。AUC仅表示事件排序，不代表总体准确率，也不是预测概率已经校准。BA加权AUC使用任务/类别权重；两类缺一则不定义。

{markdown(compact)}

每个源池跨采集监督只有12至39次有效事件，对应11至34条船，远小于查询行数。逻辑回归fit AUC约0.73至0.94；未见源船/采集的selection分别0.25、0.47、0.50、0.68，最后一池只有误伤无法算AUC。按BA权重计算后，宁波selection AUC为0.728，Shanghai为0.389；因此不能仅凭未加权事件AUC认定成功或失败。宁波有加权排序信号、Shanghai有阈值后的BA正信号，但未形成统一的跨港规律。

本轮目标函数与汇总均依次对任务内类别、采集内任务、港口内采集、港口等权。事件AUC是重复事件的描述性统计，不是独立船级泛化置信度。

固定源端选择如下，分数均是相对船体的BA增量（个百分点），平局保留船体：

{markdown(pd.DataFrame(selected_rows))}

## 第二层：完整任务是否受益

跨采集完整OA/BA及相对船体增量。random是每任务在两端预测不一致的查询中，均匀随机切换相同数量的**精确期望**，不是另选一次好种子。half是半强度船体、另一半恢复原D18项。

{markdown(result)}

source_selected不是按目标挑赢家。所有固定模型与负结果均保留。固定组合会在所有查询更新概率；学习器只在分类不同且选择启用时换端点，因此NLL/Brier不是同一连续融合操作，不能把微小差值讲成校准机制。

## 第三层：在哪里能区分，哪里不能

{markdown(target_diag)}

宁波B16/B32事件AUC虽为1.0，但只涉及8/4条事件船；其中B32的净增量与同任务同数量随机相同，说明高排序AUC也不能替代“实际选得更好”的检验。低AUC与正BA也可共存，因为阈值、类别权重和事件分布不同。

各港跨采集BA增量：

{markdown(pdeltas.loc[pdeltas.origin.eq('cross'),['pool','budget','half','mix','logistic','logistic_random','tree','source_selected']])}

### 新加坡B64局部信号

{markdown(draws.loc[draws.pool.eq('Singapore')&draws.budget.eq(64)&draws.origin.eq('cross'),['draw','mix','logistic','logistic_random','source_selected']])}

真实组合12次救回、8次误伤，逻辑回归保留全部12次救回，拦住1次误伤，剩7次。被拦住的是MMSI525901249的一个draw，分数0.465；其余得失样本详见singapore_B64_events.csv。这是值得记录的具体收益，但额外选择增量集中于一例，不能夸大为稳定边界。先前两艘重复救回船314867000、636091308的两次救回均保留，属于继承环带已有收益，不是选择器新创造。

## 根因定位与剩余不确定性

已排查：对应池的source fit/selection/target实际参与船只、采集互斥；输入不含真实查询类；改写查询标签不改变两端预测或18个输入；15组目标端点重放最大误差{max(x['max_error'] for x in verify['replay']):.3g}；救回减误伤和OA差值一致。未发现接口错误可解释这轮退化。

有数据支持的解释：

1. **真正用于学习的事件少。** 大量查询两方案完全一致或同时错误，不能靠查询总量宣称监督充足。源fit有拟合，selection普遍弱，呈现过拟合/迁移不足的表现，但尚不能把小样本确定为唯一原因。
2. **源任务与目标任务不完全同形。** 548个source fit cross任务中430个支持集混合多个采集；目标cross支持来自一个采集。source查询随draw变化，目标查询固定。该差异已核实，尚未通过匹配干预证明是损失的因果来源。
3. **源收益分布不能直接代表目标。** 新加坡源selection没有救回，目标B64却有12次救回；只按源端平均净收益选策略，会错过局部机会。这是数据中的实测反例，不是实现漏接。
4. **预算迁移尚未建立。** B16训练不等于学到了B64条件。不能用B64退化否定B16所有小收益，也不能将B16拟合解释为预算通用规则。
5. 回归系数与树规则是经验关系，不是物理定律。不同池中船体支持优势、环带类间间隔的系数方向有冲突；输入又彼此相关，不能凭系数单独作因果解释。

## 本轮决策

暂不把选择器加入主方案；保留船体局部读出的既有收益，以及新加坡等环带局部机会。没有新训练复杂门控、没有按目标重调阈值、没有展开公开集。

如果下一轮继续，最有辨别力的是**先匹配源端与目标端的支持任务组织（单采集cross、同一预算），再检验固定小探针**，而不是立刻增大网络。它能检验这里已发现的任务组成差异是否影响规律迁移；仍不保证匹配后必然有收益。新增输入或复杂模块应等这一问题厘清。

## 文件与验证

- PLAN_ZH.md：训练前冻结设计。
- SOURCE_FROZEN.json、各港_models.joblib、各港_tree.txt：源端策略及可读规则。
- source_scored.csv、target_scored.csv：逐样本输入、分数和选择；truth/gain是监督评价列，不是模型输入。
- source_ba_effects.csv、event_diagnostics_weighted.csv：训练/验证与事件区分。
- target_summary.csv、per_port_ba_effects.csv、per_port_draw_ba_effects.csv：总体、逐港和抽样结果。
- class_event_counts.csv、singapore_B64_events.csv：类别与具体船的得失。
- partitions.json、DONE.json：隔离和数值验证。

核心探针及重放运行约{verify['seconds']:.1f}秒，使用缓存CPU计算；不包含旧骨干训练/特征提取，不是端到端部署耗时。历史数据已反复开发，不作为新增独立确认；五次支持draw不是五个训练种子。
"""
    (OUT/"FINDINGS_ZH.md").write_text(text,encoding="utf-8")
    print("Wrote",OUT/"FINDINGS_ZH.md")


if __name__=="__main__":
    main()
