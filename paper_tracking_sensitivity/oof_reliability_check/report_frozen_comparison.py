"""Readable matched-comparison report, figures and existing-case gate traces."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from frozen_comparison import OUT, EXP, PARENT, load_npz, load_pickle, route_inputs
from conditional_direction_residual import fit_predict
from receiver_consistent_handoff import confidence_parts

DATASETS = ('self24ports', 'f0921', 'oss2')
NAMES = {'self24ports':'Self24 (23 tasks)', 'f0921':'F_0921', 'oss2':'OpenSARShip 2.0'}


def table(frame):
    columns = list(frame.columns)
    lines = ['| '+' | '.join(columns)+' |', '| '+' | '.join(['---']*len(columns))+' |']
    for values in frame.itertuples(index=False, name=None):
        lines.append('| '+' | '.join(f'{v:.4f}' if isinstance(v, (float, np.floating)) else str(v) for v in values)+' |')
    return '\n'.join(lines)


def case_traces():
    paths = sorted((EXP/'mechanism_20261009/cases').glob('*_draw0.json'))
    records, features, cache = [], [], {}
    for path in paths:
        case = json.loads(path.read_text(encoding='utf-8'))
        ds, task = case['dataset'], case['task']
        key = (ds, task)
        if key not in cache:
            folder = EXP/'tasks'/ds/task if ds == 'self24ports' else EXP/'tasks'/ds
            info = json.loads((folder/'PREPARED.json').read_text())
            data, old = load_npz(folder/'target.npz'), load_npz(folder/'policies.npz')
            newtrace = load_npz(OUT/'targets'/ds/task/'routing_trace.npz')
            s, sy, ev = data['support'][0], data['support_labels'][0], data['eval_masks'][0]
            score = fit_predict(data['phi'], data['prior'], data['mean'], data['conditions'], s, sy, info['eta'], conditional=True)
            gain, common, positive = route_inputs(info, data, score, data['expert'][0], sy, ev)
            candidates = np.flatnonzero(positive)
            ordered = candidates[np.lexsort((data['route_ids'][candidates], -gain[candidates]))]
            rank = np.zeros(len(ev), int)
            rank[ordered] = np.arange(1, len(ordered)+1)
            selected = (rank > 0) & (rank <= int(.05*ev.sum()))
            np.testing.assert_array_equal(selected, old['selected'][0, 0])
            cache[key] = (info, data, old, newtrace, gain, common, positive, rank, score)
        info, data, old, new, oldgain, oldcommon, oldpositive, oldrank, oldscore = cache[key]
        q = int(case['query']['local_task_row'])
        if str(data['ids'][q]) != case['query']['sample_id']:
            raise ValueError('Case ID alignment failed')
        cap = int(.05*data['eval_masks'][0].sum())
        slot = list(old['modes']).index('selective')
        e = data['expert'][0, q].argmax()
        for name, gi, co, pos, rank, bp, admitted in (
            ('previous', oldgain[q], oldcommon[q], oldpositive[q], oldrank[q], oldscore[q].argmax(), old['selected'][0,0,q]),
            ('current', new['gate_score'][0,q], new['common'][0,q], new['positive'][0,q], new['rank'][0,q],
             old['base_predictions'][0,slot,q], old['selected'][0,slot,q])):
            reason = ('same_prediction' if bp == e else 'expert_class_not_in_support' if not co else
                      'nonpositive_gate_score' if not pos else 'inside_budget' if admitted else 'rank_exceeds_budget')
            records.append(dict(dataset=ds, task=task, sample_id=data['ids'][q], event=case['event'],
                version=name, base_class=int(bp), expert_class=int(e), gate_score=float(gi),
                eligible=bool(pos), rank=int(rank), cap=cap, admitted=bool(admitted), reason=reason))
    pd.DataFrame(records).to_csv(OUT/'case_routing_explanations.csv', index=False)
    return pd.DataFrame(records)


def main():
    summary = pd.read_csv(OUT/'summary.csv')
    draws = pd.read_csv(OUT/'draw_metrics.csv')
    meta = pd.concat([pd.read_csv(OUT/f'{name}/summary.csv') for name in
                     ('r2d2_frozen_head','r2d2_normalized_head') if (OUT/f'{name}/summary.csv').exists()], ignore_index=True)
    modes = ['source_prior','prototype64','ridge_mean64','ridge_phi64','ridge_full_kernel64',
             'logistic_mean64','r2d2_frozen_head','r2d2_normalized_head','ours_base','previous_full','ours_5pct',
             'expert_all','softmax_average_all']
    allresults = pd.concat([summary, meta], ignore_index=True)
    compact = []
    for mode in modes:
        row = {'method': mode}
        for ds in DATASETS:
            r = allresults[(allresults.dataset == ds) & (allresults['mode'] == mode)]
            row[NAMES[ds]+' OA / BA'] = f'{r.iloc[0].oa:.4f} / {r.iloc[0].ba:.4f}' if len(r) else 'pending'
        compact.append(row)
    frame = pd.DataFrame(compact)
    frame.to_csv(OUT/'MAIN_COMPARISON_TABLE.csv', index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(2,3,figsize=(13.2,6.6), constrained_layout=True)
    budgets = [0,5,10,20,50,100]
    for col, ds in enumerate(DATASETS):
        for ri, metric in enumerate(('oa','ba')):
            ax = axes[ri,col]
            subset = summary[summary.dataset == ds].set_index('mode')
            names = [f'budget_{b:03d}' for b in budgets]
            values = subset.loc[names,metric].to_numpy()
            sd = draws[(draws.dataset == ds) & draws['mode'].isin(names)].groupby('mode')[metric].std().reindex(names).to_numpy()
            ax.plot(range(6),values,'o-',color='#167d9a',label='Budgeted handoff')
            ax.fill_between(range(6), values-sd, values+sd, color='#167d9a',alpha=.12)
            ax.axhline(subset.loc['expert_all',metric],color='#a64c4c',ls='--',label='All expert')
            ax.axhline(subset.loc['softmax_average_all',metric],color='#72723b',ls=':',label='All-query fusion')
            ax.set_xticks(range(6),budgets)
            ax.set_xlabel('Handoff cap (%)')
            ax.set_ylabel(metric.upper()+' (%)')
            ax.set_title(NAMES[ds])
            ax.grid(axis='y',alpha=.2)
    axes[0,0].legend(fontsize=8,loc='best')
    fig.savefig(OUT/'budget_comparison.png',dpi=180)
    fig.savefig(OUT/'budget_comparison.pdf')
    plt.close(fig)
    cases = case_traces()
    routes = summary[summary['mode'].isin(['ours_5pct','random_common05','margin_common05','entropy_common05',
                     'random_positive_matched','margin_positive_matched'])][['dataset','mode','oa','ba','handoff_rate']]
    budget_table = summary[summary['mode'].str.startswith('budget_') | summary['mode'].isin(['expert_all','softmax_average_all'])][
        ['dataset','mode','oa','ba','handoff_rate','expert_decision_use_rate']]
    text = '# 冻结主模型的匹配对比实验\n\n日期：2026-10-10。第一批23模式、25任务、125组支持抽样、2875条任务级结果全部完成。'
    text += '原主模型基础/最终预测与5%交接名单逐项复现。此处的模式数包括重复一致性参考行，不等于23个独立公开算法。\n\n'
    text += '## 结论\n\n当前5%交接在三个集合的平均OA/BA均优于同预算随机、低margin与高entropy交接，也优于更严格的相同正收益资格集合/相同实际交接数量对照。'
    text += '但不限预算的简单双分支softmax均值融合在三个集合的OA/BA都更高。因此可以说有限专家接管下排序有效，不能说5%系统的绝对精度最高。'
    text += '当前已经全量计算两路，尚无真实计算节省证据；预算的部署含义必须有实际应用依据，不能用虚构成本回避融合强基线。主版本不根据这些目标成绩重新选择预算或模块。\n\n'
    text += '## 完整识别比较\n\n'+table(frame)+'\n\n'
    text += 'source_prior为零目标标签参考。其余头共享同64标签；ridge/proto/logistic是标准强基线，不是完整公开深度模型复现。'
    text += 'mean为冻结C4的源标准化PCA128单位向量；phi包含数据集特定几何；full_kernel读取与主方案相同学习核但不带源分数锚点。三者的信息与先验访问不同，分别标注而非混作单变量消融。'
    text += '岭回归正则、逻辑回归C与原型距离在既有源情景上选定，目标不选参数；源端表示本身未与这些情景完全隔离。'
    text += '原型/普通头仅输出支持中观察到的类，源先验方法能保留八类先验。OSS2有普通头OA高于我们而BA低于我们的情况，不能写成全部指标全面胜出。\n\n'
    text += 'R2D2行是明确的冻结特征协议适配：通过源查询交叉熵反传岭回归求解器，学习正则和分数尺度；不是普通网格ridge，也不是端到端R2D2完整复现。'
    text += '固定30轮、末轮输出，没有目标端选轮数；八类自然支持分布不同于原论文均衡N-way/K-shot。此行不用于宣称战胜原论文完整方法。\n\n'
    text += '两条元ridge结果均保留：frozen_head是不带截距的初版启发式头；normalized_head补齐官方公式中的常数列和平均平方误差尺度。'
    text += '后者初始lambda=1/64，使64支持样本下有效求和正则仍为1；这是单位尺度转换，不是目标选参。alpha仍初始1，30轮/Adam .01不变。'
    text += '官方默认lambda50且不学习，原发布配置alpha可为1e-4或10；本实验不使用这些默认训练设置，也不重训骨干，仍只称冻结特征协议适配。\n\n'
    text += '对齐公式后BA没有改善的结果同样保留。类别表显示，该适配在F_0921对渔船/杂货船/近海供应船的召回仅约1.69/1.86/0.85%，'
    text += '在OSS2对成品/化学品油轮、集装箱船则高于当前主模型，而对拖轮/近海供应船召回约1.10/0.75%。因此高OA与低BA确实对应类别取舍。'
    text += '已验证公式与独立原始空间解、梯度及官方平均损失形式一致；但未证明训练已达到最优，不能把差距全部归因于损失不平衡或据此否定完整R2D2。两版本class_metrics/class_summary已保存。\n\n'
    text += '## 交接策略比较\n\n'+table(routes)+'\n\n'
    text += 'common05使用共同的查询/预测分歧/专家类别已观察条件，不依赖我方收益分数符号。positive_matched进一步固定我方正收益资格集合和实际数量，只比较排序。'
    text += '随机每支持抽样一个预定随机数种子，共5个，不是独立的大规模随机策略重复。软最大分数只是排序/融合运算，不宣称已校准概率。\n\n'
    text += '## 当前版本预算曲线\n\n'+table(budget_table)+'\n\n'
    text += '100%门控仍有资格与正收益过滤，不等于全专家。全量融合不属于硬交接，handoff_rate记为空，expert_decision_use_rate为100%；后者是专家输出参与最终决策比例，不是特征计算比例。'
    text += '5%不是普遍最优点，放宽预算也非单调改善BA：OSS2到50%时BA低于5%。不按目标结果事后修改主工作点。\n\n'
    text += f'![Budget curves]({(OUT/"budget_comparison.png").as_posix()})\n\n阴影为五组支持抽样指标的标准差，不是独立训练置信区间。横轴是离散预算档位；虚线参考使用不限预算专家信息。\n\n'
    text += '## 可解释性同步补齐\n\n'+table(cases[['dataset','task','sample_id','version','gate_score','rank','cap','admitted','reason']])+'\n\n'
    text += '已复用原六张案例，补齐旧/新交接分数、资格、排名、截止数和实际原因。rank=0表示未进入资格集合，不表示排名第一。'
    text += '原预测名单重放一致；这些是计算决策轨迹，不是物理散射解释。25个任务、共125组的全量轨迹在targets/**/routing_trace.npz。\n\n'
    text += '## 尚未完成的比较\n\n完整端到端公开SAR分类强方法与完整公开learning-to-defer方法尚未复现。R2D2冻结头只填补对应适配基线，不能替代这两项。'
    text += '公开方法核读及可复现性见PUBLIC_METHODS_ZH.md（该支线归档后生效）。本报告不声称完整SOTA比较已经齐备。\n\n'
    text += 'SARATR-X可执行性复查已完成：本地有官方源码，现有CUDA环境可导入HiViT；约1.044GB官方权重匿名401、现有凭据403，未获下载授权。'
    text += '已向用户询问可用权重路径；不绕过门控，不拿随机初始化模型冒充该方法。官方预训练包含OpenSARShip，具体目标重叠仍待核实。\n\n'
    text += '## 文件\n\nsummary.csv、metrics.csv、draw_metrics.csv、class_metrics.csv、MAIN_COMPARISON_TABLE.csv、case_routing_explanations.csv、PROTOCOL.json；'
    text += 'r2d2_frozen_head/单独保存适配方法协议、源训练参数、目标预测与结果。全部结果是历史开发数据，同五组支持抽样，不是五训练种子；原实验工件未覆盖。\n'
    (OUT/'FINDINGS_ZH.md').write_text(text,encoding='utf-8')
    print(frame.to_string(index=False),flush=True)
    print('Report, PDF/PNG figures and six-case routing trace exported.',flush=True)


if __name__ == '__main__':
    main()
