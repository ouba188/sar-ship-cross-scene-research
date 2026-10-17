"""Report matched-source probes, separating learned policies from fallback rows."""

import json

import joblib
import numpy as np
import pandas as pd

from freeze_ring_matched import OUT
from probe_ring_predictability import OUT as PREVIOUS, aggregate, risk_weights, save_json
from report_ring_predictability import markdown


def main():
    coverage=pd.read_csv(OUT/"learning_coverage.csv")
    tasks=pd.read_csv(OUT/"task_coverage.csv")
    ports=pd.read_csv(OUT/"target_per_port.csv")
    decisions=pd.read_csv(OUT/"target_decisions.csv",dtype={"mmsi":str})
    metrics=pd.read_csv(OUT/"target_metrics.csv")
    events=pd.read_csv(OUT/"event_diagnostics.csv")
    availability=pd.read_csv(OUT/"target_availability.csv")
    done=json.loads((OUT/"DONE.json").read_text(encoding="utf-8"))

    delta=ports.pivot(index=["recipe","pool","budget","origin"],columns="arm",values="ba")
    delta=delta.subtract(delta.body,axis=0).reset_index()
    delta.to_csv(OUT/"per_port_ba_deltas.csv",index=False)
    arms=["body","mix","half","logistic","logistic_random","tree","source_selected"]
    available=availability.loc[availability.recipe.eq("single_budget")&availability.trained,["pool","budget"]]
    comparison=delta.loc[delta.origin.eq("cross")].merge(available,on=["pool","budget"],validate="many_to_one")
    comparison.to_csv(OUT/"trained_scope_recipe_comparison.csv",index=False)
    single=delta.loc[delta.recipe.eq("single_budget")&delta.origin.eq("cross")].merge(
        coverage.loc[coverage.organization.eq("single")],on=["pool","budget"],validate="one_to_one")
    single.loc[~single.trained,["logistic","logistic_random","tree","source_selected"]]=np.nan
    single.to_csv(OUT/"matched_results_with_missing_as_na.csv",index=False)
    matched_scope=comparison.groupby(["recipe","budget"])[arms].mean().reset_index()
    matched_scope.to_csv(OUT/"common_trained_scope_summary.csv",index=False)
    draw_frames=[]
    for recipe,g in metrics.groupby("recipe"):
        a=aggregate(g,["draw"])
        a["recipe"]=recipe
        draw_frames.append(a)
    draws=pd.concat(draw_frames,ignore_index=True)
    draw_delta=draws.pivot(index=["recipe","pool","budget","origin","draw"],columns="arm",values="ba")
    draw_delta=draw_delta.subtract(draw_delta.body,axis=0).reset_index()
    draw_delta.to_csv(OUT/"per_port_draw_ba_deltas.csv",index=False)

    focus=decisions.loc[decisions.recipe.eq("single_budget")&decisions.pool.eq("Jebel Ali")&
        decisions.budget.eq(16)&decisions.origin.eq("cross")].copy()
    focus.to_csv(OUT/"jebel_B16_decisions.csv",index=False)
    weights=risk_weights(focus)
    focus["ba_contribution_pp"]=100*weights*focus.gain*focus.choose_logistic
    focus["mix_ba_contribution_pp"]=100*weights*focus.gain
    focus["rescue"]=focus.gain.eq(1)&focus.choose_logistic
    focus["harm"]=focus.gain.eq(-1)&focus.choose_logistic
    class_table=focus.groupby("truth").agg(query_samples=("sample_id","nunique"),
        query_ships=("mmsi","nunique"),rescues=("rescue","sum"),harms=("harm","sum"),
        selector_ba_contribution_pp=("ba_contribution_pp","sum"),
        mix_ba_contribution_pp=("mix_ba_contribution_pp","sum")).reset_index()
    class_table.to_csv(OUT/"jebel_B16_class_contributions.csv",index=False)
    focus_units=metrics.loc[metrics.recipe.eq("single_budget")&metrics.pool.eq("Jebel Ali")&
        metrics.budget.eq(16)&metrics.origin.eq("cross")]
    scene=focus_units.groupby(["acquisition","arm"])[["oa","ba"]].mean().unstack("arm")
    scene_delta=pd.DataFrame({"acquisition":scene.index,
        "selector_oa_gain":(scene.oa.logistic-scene.oa.body).to_numpy(),
        "selector_ba_gain":(scene.ba.logistic-scene.ba.body).to_numpy(),
        "vs_random_ba":(scene.ba.logistic-scene.ba.logistic_random).to_numpy(),
        "vs_mix_ba":(scene.ba.logistic-scene.ba.mix).to_numpy()})
    scene_delta.to_csv(OUT/"jebel_B16_by_scene.csv",index=False)
    np.testing.assert_allclose(class_table.selector_ba_contribution_pp.sum(),scene_delta.selector_ba_gain.mean(),atol=1e-10)
    save_json(OUT/"CANDIDATE.json",dict(model_file=str(OUT/"models.joblib"),
        model_key=["Jebel Ali","single",16],learner="logistic",threshold=.5,
        status="retain_scoped_candidate_not_main_model",scope="Jebel Ali cross-acquisition B16",
        source_selection_available=False,scope_discovery="historical_target_development_results",
        delta_oa_pp=float(scene_delta.selector_oa_gain.mean()),delta_ba_pp=float(scene_delta.selector_ba_gain.mean()),
        additional_ba_vs_fixed_mix_pp=float(scene_delta.vs_mix_ba.mean()),
        additional_ba_vs_same_count_random_pp=float(scene_delta.vs_random_ba.mean())))

    model=joblib.load(OUT/"models.joblib")
    constants=[]
    for (pool,org,budget),m in model.items():
        if not m:
            continue
        scaler,logistic=m["logistic"]
        assert np.isfinite(scaler.scale_).all() and (scaler.scale_>0).all()
        assert np.isfinite(logistic.coef_).all()
        constants.append(dict(pool=pool,organization=org,budget=budget,tree_nodes=m["tree"].tree_.node_count,
            min_raw_scaler_variance=float(scaler.var_.min())))
    pd.DataFrame(constants).to_csv(OUT/"model_diagnostics.csv",index=False)
    assert np.isfinite(decisions[["score_logistic","score_tree"]].to_numpy()).all()
    save_json(OUT/"REPORT_CHECKS.json",dict(scaler_and_model_outputs_finite=True,
        tiny_negative_variance_min=min(r["min_raw_scaler_variance"] for r in constants),
        constant_trees=sum(r["tree_nodes"]==1 for r in constants),
        fitted_tree_count=len(constants),focus_class_contributions_sum_to_ba=True,
        unavailable_matched_models_reported_as_na=True))

    compact=coverage.loc[coverage.organization.eq("single"),["pool","budget","fit_query_scenes","fit_rescues",
        "fit_harms","fit_event_ships","selection_query_scenes","selection_rescues","selection_harms","trained"]]
    focus_draw=draw_delta.loc[draw_delta.recipe.eq("single_budget")&draw_delta.pool.eq("Jebel Ali")&
        draw_delta.budget.eq(16)&draw_delta.origin.eq("cross"),["draw","logistic","logistic_random","mix"]]
    focus_compare=comparison.loc[comparison.pool.eq("Jebel Ali")&comparison.budget.eq(16),
        ["recipe","logistic","logistic_random","mix","half"]]
    focus_event=events.loc[events.pool.eq("Jebel Ali")&events.budget.eq(16)&events.origin.eq("cross")&
        events.model.eq("logistic"),["organization","split","rescue","harm","event_ships","auc","selected_rescue","selected_harm"]]
    raw_effect=single[["pool","budget","trained","status","logistic","logistic_random","mix","half"]]
    text=f"""# 匹配源支持组织与预算：复测结果

## 结论

**匹配后出现一个明确值得保留的局部正信号：Jebel Ali、16个支持。并未得到全港、全预算均有效的选择器；更大的限制是当前合法源任务覆盖不足。**

- 该任务逻辑回归较船体BA +{scene_delta.selector_ba_gain.mean():.4f}个百分点、OA +{scene_delta.selector_oa_gain.mean():.4f}个百分点；五次支持draw的BA增量全部为正。
- 较固定环带组合BA +0.1819，较同任务同切换数量随机期望 +0.2005；本轮不是仅仅“少用一些环带”。五次draw较随机期望都为正，但这些是同一查询队列的支持重复抽样，不是五次独立训练。
- 固定组合15次救回、10次误伤；选择器保留14次救回、剩5次误伤，即拦住5次误伤、错过1次救回。14次救回涉及12条船，5次误伤涉及5条船（两集合可能同船在不同draw出现）。
- 同样源query/支持类配额下，多采集训练对照只有+0.1001 BA；旧训练方案为+0.2248。支持组织匹配对该局部任务有实测支持，但没有据此宣称它是所有失败的唯一原因。
- 该池没有合格的独立source selection任务，因此这里是**预先固定C和0.5阈值的探针结果**，不是源验证成功选出的部署规则。目标结果未用于调参数，发现的港口适用范围仍属探索性。

## 本轮实际完成的匹配

源端单采集支持、固定查询、每类配额、same/cross共同容量、嵌套B16/B32/B64、五次draw、船只与几何排重均匹配目标组织。沿用原source split，不把selection借作fit，也没有合并五池。每个合法pool/budget分别训练原18特征逻辑回归和二层树；阈值、正则、树深度和叶节点约束不变。

新增混合采集对照与单采集任务共享query、预算、逐类数量；实际能否取到多个采集列于task_coverage.csv。Qingdao等部分控制任务只能来自一个合法采集，不能把这些任务算作真正多采集干预。

冻结840个任务（包括same/cross、两种组织、fit/selection、预算和draw重复），产生{done['source_rows']}行源端记录，实际拟合{done['new_trained_models']}个探针。目标仍使用上轮完全相同的查询、支持和两端概率，不新增公开数据。

## 匹配覆盖必须如实看

下表是single源端的实际覆盖。fit_query_scenes是源query的港口-采集场景数；事件包含五次支持抽样的重复观察。trained只表示两类事件均存在并完成拟合，不表示有独立匹配验证，更不表示效果通过。

{markdown(compact)}

实际可训练single探针：B16三个池（Jebel、Qingdao、Shanghai），B32两个池（Jebel、Shanghai），B64一个池（Jebel，仅3个事件/3条船）。宁波B16仅1次误伤，新加坡B16零事件，无法训练二类收益选择器。不能把这些格子的body回退记成“已匹配且没有损失”。

有single训练并有matching selection任务的只有Qingdao B16与Shanghai B16，后者selection只有一个有效事件。Jebel三个预算和Shanghai B32均无匹配selection；源选择策略按预先约定回退body，不拿目标挑策略。

## 各任务结果，不把缺模型算胜利

以下均是相对船体的跨采集BA百分点。NA表示没学成匹配模型，非零收益或零退化。B64 Jebel的0是模型确实训练后在全部42个分歧查询上选择关闭，其他池B64的NA则是根本没有该预算模型，两者不能混淆。

{markdown(raw_effect)}

same结果、源端选择结果、NLL/Brier及所有负结果保留在target_metrics.csv。target_summary_with_fallback.csv只表示带body回退的操作结果，不可当作全覆盖学习效果主表。主比较应使用matched_results_with_missing_as_na.csv及common_trained_scope_summary.csv。

## Jebel Ali B16：究竟改善在哪里

### 同目标队列下的训练配方对照

{markdown(focus_compare)}

single_budget与mixed_budget共用源查询、预算和类配额，因此比较比旧训练与新训练的直接差值更有辨别力。不同支持组织会改变支持船组成和救回/误伤监督，这是该组织干预的作用路径，并不等价于证明某种物理散射机制。

### 五次支持抽样

{markdown(focus_draw)}

### 四个采集

{markdown(scene_delta)}

3/4采集的BA较船体为正，另一个为负；这是适用边界，不否定整体净收益。相对同数量随机，3/4采集更好。

### 类别贡献

{markdown(class_table)}

贡献按相同任务内类别、采集和港口权重计算，可加总还原完整BA增量；不是各类别BA变化直接相加。类别编号沿用原八类映射，个别类别缺失不会虚构样本。

### 是否学到了可区分的关系

{markdown(focus_event)}

该目标队列事件排序AUC从旧探针0.4667提高到0.7533；对应14次救回、5次误伤。这同时有排序与净收益证据，不只看训练拟合。不过事件只有22条不同船，且无本池匹配source selection，不能叫已充分确认的跨港通用规律。

回归主要利用船体/环带的类支持一致性、局部删支持稳定性、支持类别占比等组合。ring_loo_stability在该模型中系数为负，说明不能把结果预先讲成“环带越稳定越值得信任”。这些相关输入的联合规则不等于单项因果解释；系数文件已完整保存。

## 为什么大预算没有补齐

独立只读检查确认，问题并非把预算参数改成64就能解决：需要同港至少两次采集，扣除固定query、同船与重叠裁片后，仍有足够逐类共同支持；fit与selection还必须互不共享船和采集。

- **Jebel：** source selection的Hamburg两采集逐类共同容量最多15，连B16都组不出。fit中Port Klang一方向排重后same58不足B64，反方向可用，所以只剩一个B64场景。
- **Qingdao：** 三个大Singapore采集被旧split拆成一大采集在fit、两大采集在selection，存在明确碎片化。但总共三个大采集，仍组不出两组采集互斥的B64配对。
- **Ningbo：** Busan、Tanger Med配对被split拆散，重新按配对划分有机会增加B32。恢复历史被剔除的validation记录只给出B64乐观上限，当前固定query排重后same60，不能称为已可用64。
- **Shanghai、Singapore：** 目前合法源validation即使恢复遗漏记录并暂忽略split，逐类共同容量也不足以支持B64。不能靠合池或从validation借船补齐。

因此这一轮还不能回答“全五池同预算匹配后是否都能学到规律”。已完成的是当前合法覆盖上的匹配对照，以及覆盖不足的具体定位。

## 根因判断与保留方案

1. **可保留：** Jebel B16的单采集训练选择器，固定设计下五次支持draw净BA均为正，且平均超过固定组合、半强度船体和同数量随机。它是局部候选，不直接替换全局主模型。
2. **不能推广：** Qingdao B16仍退化；Shanghai B16只是继承一个环带救回，未超随机；两个匹配B32均无净提升；B64只有3事件学习后全关闭。
3. **已排除的实现问题：** 新任务索引、船只/采集隔离、预算嵌套与配额，查询标签改写不改变预测/特征，以及旧目标端点/旧选择器重放均通过。未发现需要修代码才能解释的退化。
4. **下一步若继续：** 在不改变模型的前提下，优先按合法采集配对组织源fit/selection，补齐B16/B32可学习事件；B64需要更多合法源配对采集覆盖，不能仅重排当前缓存就声称完成。先冻结本轮Jebel候选，不依据目标结果再调C或阈值。

## 技术记录

小探针及目标缓存评价运行约{done['seconds']:.1f}秒，不含既有骨干/特征提取。14棵树中12棵为根节点常量预测，符合未改变的min_samples_leaf=10；不能把它们写成学到了条件分支。

Ningbo混合B16的标准化曾出现浮点方差-2.19e-47的警告；常量特征实际scale被设为1，所有模型参数、尺度和输出均为有限数，未发现传播NaN。这是已定位的数值舍入，不用于解释任务得失，也没有据结果更换实现。

文件：frozen.json/PLAN_ZH.md记录先验设计；learning_coverage.csv和SOURCE_FROZEN.json记录可学习性/选择；models.joblib、logistic_coefficients.csv与tree.txt保存模型；per_port_ba_deltas.csv、per_port_draw_ba_deltas.csv、jebel_B16_class_contributions.csv保存细分结果；DONE.json与REPORT_CHECKS.json记录验证。

数据仍是历史开发数据，不是新独立确认；固定模型五次支持抽样不等于五个训练种子。相似模型或负结果不否定局部收益，但也不把未训练格子的回退结果包装成成功。
"""
    (OUT/"FINDINGS_ZH.md").write_text(text,encoding="utf-8")
    print("Wrote",OUT/"FINDINGS_ZH.md")


if __name__=="__main__":
    main()
