"""Describe the fixed baseline results without fitting or selecting another model."""
import json
import numpy as np
import pandas as pd
from run_baseline import HERE, load, metrics


def main():
    f,raw=load()
    p=pd.read_csv(HERE/'predictions.csv',dtype={'mmsi':str})
    p=p[p.role.eq('test')].copy()
    p['vh_valid']=raw['valid'][p.row_index,1]
    p['correct']=p.prediction.eq(p.class_id)
    p['month']=p.time.str[:7]
    pd.crosstab(p.class_id,p.prediction).reindex(index=range(8),columns=range(8),fill_value=0).to_csv(HERE/'confusion_counts.csv')
    records=[]
    for key,part in p.groupby('vh_valid'):
        records.append(dict(vh_valid=key,**metrics(part.class_id.to_numpy(),part[[f's{c}' for c in range(8)]].to_numpy())))
    pd.DataFrame(records).to_csv(HERE/'vh_metrics.csv',index=False)
    p.groupby(['class_id','class_name','vh_valid']).agg(n=('correct','size'),OA=('correct',lambda x:x.mean()*100)).reset_index().to_csv(HERE/'class_vh_metrics.csv',index=False)
    p.groupby('month').agg(n=('correct','size'),OA=('correct',lambda x:x.mean()*100)).reset_index().to_csv(HERE/'month_metrics.csv',index=False)
    s=json.loads((HERE/'summary.json').read_text(encoding='utf-8'))
    v=json.loads((HERE/'verification.json').read_text(encoding='utf-8'))
    cm=pd.read_csv(HERE/'class_metrics.csv')
    cm=cm[cm.fold.eq('pooled')]
    cov=pd.read_csv(HERE/'cohort_coverage.csv')
    lines=['# 八类冻结 baseline：结果与证据边界','',
        f'完成：新 frozen layer3/PCA128/Ridge8 在 {len(p)} 条互斥外层测试上的 pooled OA **{s["pooled"]["OA"]:.3f}%**、BA **{s["pooled"]["BA"]:.3f}%**。这不是旧 A+D，也不是此前三类 strong reader 的同一 baseline。没有训练 attention。','',
        '## 固定协议结果','',
        '| 外层 | train / val / test | alpha | test OA | test BA | source OOF OA / BA |','|---|---:|---:|---:|---:|---:|']
    met=pd.read_csv(HERE/'metrics.csv')
    coverage=pd.read_csv(HERE/'split_coverage.csv')
    for fold in range(3):
        m=met[(met.fold==fold)&met.role.eq('test')].iloc[0]
        o=met[(met.fold==fold)&met.role.eq('source_oof')].iloc[0]
        c=coverage[(coverage.fold==fold)&coverage.inner.eq(-1)].set_index('role')
        lines.append(f'| {fold} | {c.loc["train","n"]} / {c.loc["validation","n"]} / {c.loc["test","n"]} | {m.alpha:g} | {m.OA:.3f} | {m.BA:.3f} | {o.OA:.3f} / {o.BA:.3f} |')
    lines += ['',
        '三折 alpha=1 都由9月验证集 balanced NLL 选择；未根据测试结果更换表示、超参数或切分。候选 NLL 差异较小，原值见 alpha_candidates.csv，不将 alpha=1 解读为已证明的机制。',
        '三个 source OOF 共13229个折内预测位置；同一观察可能属于多个外层 source，这不是13229个独立样本。外层 test 为4744个唯一观察、2973个身份，覆盖父队列全部 >=2025-10-01 的观察，每条恰评价一次。父队列15251条仍全量保留在每折缓存，不为了测试覆盖额外分配其余时间/身份组合。','',
        '## 八类与有效观测','',
        '| 类别 | 全队列 n | 全队列有效 VH | test n | test recall % | test有效 VH |',
        '|---|---:|---:|---:|---:|---:|']
    for r in cm.itertuples():
        c=cov[cov.class_id.eq(r.class_id)].iloc[0]
        lines.append(f'| {r.class_name} | {c.n} | {c.vh_valid} | {r.n} | {r.recall:.3f} | {r.vh_valid} |')
    lines += ['',
        f'总计 {s["rows"]} 条、{s["mmsi"]} MMSI、{s["products"]} 产品、{s["passes"]} 过境；VV全部有效，VH {s["vh_valid"]}/{s["rows"]} ({100*s["vh_valid"]/s["rows"]:.2f}%)；测试 VH {s["test_vh_valid"]}/{len(p)} ({100*s["test_vh_valid"]/len(p):.2f}%)。1468条无效VH保留有效VV，没有整体清掉样本。',
        '当前读出最薄弱的是 tug_towing (recall 9.901%)；product/chemical tanker 虽有1480条测试但 recall 31.014%，因此低总分不能只归结为小类数量不足。fishing 的全队列有效VH仅674/1427，测试76/145，却有66.207% recall；缺失VH与类别组成相关，不能从总体分组OA直接推出缺失的因果损失。',
        '12个 inner fit 均覆盖八类，最小单类 fit 为36条；validation 的 tug 最少7条、各测试 fishing 41~58条，逐类估计仍有有限样本边界。按身份加权不等于消除了数据组成或时间迁移。','',
        '## 固定线性与缓存验证','',
        f'- 3个 outer、12个 inner 模型落盘；含候选共21次Ridge拟合，0个 attention 训练。baseline与条件投影合计运行 {s["seconds"]:.2f} 秒；独立缓存重放与四图生成 {v["seconds"]:.2f} 秒，不含人工阅图/写报告。',
        f'- 完整 q+b/base 与 oq+ob/oof 最大分解误差 {s["max_decomposition_error"]:.3g}；15个模型独立序列化重放最大误差 {v["max_linear_error"]:.3g}，条件投影采样重放误差 {v["max_projection_error"]:.3g}（float32矩阵批量舍入）。',
        '- 原float32管线与float64线性评价 top1 差异0；全部保存的浮点缓存有限、全队列ID/顺序一致。OOF非source位置为0+false；每条source恰有一个held预测。',
        '- W为8x2048，16组严格为原layer3通道VV8组+VH8组；保存无权h0、训练加权h0及对应b。不能以outer b重建OOF，应使用逐行ob和各inner自己的W/h0。',
        '- 条件投影全程不使用标签；outer/inner各自在其fit的whole stem上拟合PCA64及前后标准化。whole64/fg64和双区均按固定sqrt32尺度；双区前32来自同一PCA64，不混淆原读出分组与PCA轴。',
        '- 无效VH原块为0，但PCA标准化会让缺失状态产生坐标偏移；保留valid/regional_valid供后续使用。未补VH，也未将有VV的整条向量置零。','',
        '## 已实际回看的原图','',
        '四例按协议固定类别及obs_id首例选择，不按大分差挑选；均为真实外层test、VV/VH分别显示原160 crop/pad与实际mask。已使用view_image逐张查看四图，并查看父region_coverage.png的八类面板。每例路径、obs_id、fold、预测和八类分数完整记录于review_cases.csv。',
        '',
        '| 状态 | instance_id / obs_id / fold | 证据变化与可见事实 |',
        '|---|---|---|',
        '| 正确 bulk | returned_20260915_0001878 / 74 / 1 | 真类分差+0.222945；VV/VH均为斜向较连续亮区，OBB覆盖主体，原有效区域外围有大量padding。判对不证明用了背景。 |',
        '| 正确 container | returned_local_knowledge_0020310 / 205 / 2 | 真类分差+0.220151；横向多个强亮响应，VV可见延伸亮线，VH也有分离强点；与bulk例外观不相同，但不把亮点排列直接命名为确定船型结构。 |',
        '| 错误 product/chemical -> bulk | returned_local_knowledge_0005897 / 403 / 0 | 真类分差-1.252875；VV是小亮簇，VH有更大的有效画布、明显十字延伸及更纵向的OBB。两pol均按父标记有效；此图直接显示输入观测差异，不能据此擅改有效标记或认定其就是误分类根因。 |',
        '| 错误 general cargo -> offshore supply | returned_20260915_0004141 / 222 / 2 | 真类分差-0.319727；VV/VH目标都只占很小的中心亮区，背景细节有限，OBB覆盖亮主体。有限形态信息可能造成类间混淆，但单图不能证明缺失某种上下文即可救回。 |',
        '',
        '这里仅有一个baseline，没有模型间翻转，故按要求用正确/错误各两例代替救回/误伤/CC/WW，不能把案例当成全体比例。两例错误VH均有效，故“全部错误由缺失VH造成”不成立；但不能因此量化排除缺失对其他样本的影响。',
        '全图深层均值读出、PCA降维、八类相似外观、身份/日期迁移及不同chip有效画布都可能影响区分；本轮只能确认链路符合固定设计、实际混淆集中在哪些类别及这些原图现象，没有分别操控这些解释，不能选定唯一根因。此处也不声称某attention或区域组织必然补上差距。','',
        '## 交付与使用','',
        '- run_baseline.py：训练指定baseline及OOF；verify_and_review.py：独立落盘重放和原图面板；analyze_results.py：只读已有预测生成分析。',
        '- fold0_inputs.npz、fold1_inputs.npz、fold2_inputs.npz：全15251行缓存，含base/oof/q/oq/b/ob、W/h0、三种条件投影及对应OOF、有效标记、split下标。',
        '- models/：15个本地线性/预处理参数文件；predictions.csv、oof_predictions.csv、metrics.csv、pooled.csv、class_metrics.csv。',
        '- split_membership.csv、split_coverage.csv、pass_assignment.csv、alpha_candidates.csv、linear_checks.csv、serialized_replay.csv、verification.json、summary.json；confusion_counts.csv、class_vh_metrics.csv、vh_metrics.csv、month_metrics.csv、acquisition_metrics.csv。',
        '- 具体数组和精度规则见PROTOCOL_ZH.md。父特征/标签/原图均未修改，不提新特征、不训练新attention、不Git。',
        '']
    (HERE/'FINDINGS_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
    print(pd.DataFrame(records).to_string(index=False))
    print('Report written:',HERE/'FINDINGS_ZH.md')


if __name__=='__main__':
    main()
