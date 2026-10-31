"""Historical spatial-M3 rescue: frozen replay, exact writeback interventions, figures."""
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HISTORY = ROOT.parent / 'm3_explained_20261029'
sys.path.insert(0, str(HISTORY))
import trace_relations as tr
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from matplotlib.patches import Rectangle, ConnectionPatch

POL = tr.POL
COLORS = (tr.GREEN, tr.BLUE, tr.ORANGE)
REGIONS = tr.REGIONS


def save(fig, name):
    fig.savefig(ROOT / f'{name}.png', dpi=170, bbox_inches='tight', facecolor='white')
    fig.savefig(ROOT / f'{name}.pdf', bbox_inches='tight', facecolor='white')
    tr.plt.close(fig)


def picture(ax, view, title, chosen=None):
    if view is None:
        ax.text(.5, .5, '该极化缺失\n不参与该极化匹配', transform=ax.transAxes,
                ha='center', va='center', fontsize=10)
        ax.set_title(title, fontsize=10); ax.set_axis_off()
        return None
    ax.imshow(view['pixels'], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    masks = tr.exp.tokens.slots(view['geometry'], view['footprint'])
    for slots, color in zip(tr.exp.tokens.REGIONS, COLORS):
        mask = masks[slots].sum(0)
        if mask.max() > .05:
            ax.contour(mask, levels=[.05], colors=[color], linewidths=.65)
    if chosen is not None:
        ax.contour(masks[chosen], levels=[.05], colors=['#B2188B'], linewidths=1.3)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    return masks


def rowscore(name, score):
    score = tr.arr(score)[0]
    return dict(condition=name, prediction=int(score.argmax()),
                predicted_class=tr.CLASSES[score.argmax()],
                margin=float(tr.review.margin(score[None], np.array([0]))[0]),
                bulk_logit=float(score[0]), chemical_logit=float(score[3]), logits=score.tolist())


@torch.no_grad()
def main():
    exp, joint = tr.exp, tr.j
    exp.DEVICE = 'cpu'; exp.current.setup('cpu'); torch.set_num_threads(2)
    rows, nf, refs, rid, arrays = exp.load()
    cache = joint.pair_cache()
    idx = next(i for i, row in enumerate(rows) if str(row['id']) == '12457')
    model = tr.review.restore('N_level_pattern', 17)
    model.requires_grad_(False)
    full, full_details = joint.forward(model, 'N_level_pattern', cache, arrays,
                                      np.array([idx]), rid, rows, details=True)
    for op in model.m3[0]:
        op.frequency = False
    spatial, details = joint.forward(model, 'N_level_pattern', cache, arrays,
                                    np.array([idx]), rid, rows, details=True)
    original = exp.tensor(arrays['x'], [idx])
    updates = torch.stack([d['update'] for d in details], 1)
    combined = original + updates
    reconstructed = tr.downstream(model, arrays, idx, combined)
    absent = tr.downstream(model, arrays, idx, original)
    saved = np.load(HISTORY / 'relation_evidence/12457_traces.npz')['intervention_logits']
    checks = dict(full_replay_error=float(abs(tr.arr(full)[0] - saved[0]).max()),
                  spatial_replay_error=float(abs(tr.arr(spatial)[0] - saved[1]).max()),
                  downstream_error=float((reconstructed - spatial).abs().max()))
    assert all(v < 1e-4 for v in checks.values()), checks
    assert int(spatial.argmax()) == 0 and int(absent.argmax()) != 0
    conditions = [rowscore('M3回写关闭', absent), rowscore('空间M3开启（主图）', spatial),
                  rowscore('历史空间＋频域M3开启', full)]
    ids = [str(rows[i]['id']) for i in rid]
    same, other = ids.index('31512'), ids.index('13000')
    interventions = []
    for pol in range(2):
        component, error = tr.decompose(model, pol, details[pol], arrays, idx, rid)
        checks[f'decomposition_{POL[pol]}'] = error
        cases = [(f'{POL[pol]}全部回写', component())]
        cases += [(f'{POL[pol]}{REGIONS[r]}回写', component(slots=s))
                  for r, s in enumerate(exp.tokens.REGIONS)]
        cases += [(f'{POL[pol]}参考{ids[k]}回写', component(refs=[k])) for k in (same, other)]
        for name, update in cases:
            modified = combined.clone(); modified[:, pol] -= update
            record = rowscore('移除' + name, tr.downstream(model, arrays, idx, modified))
            record['margin_change'] = record['margin'] - conditions[1]['margin']
            interventions.append(record)
    # Independent zero-gain execution detects mistakes in the bypass calculation.
    for op in model.m3[0]:
        op.gain.zero_()
    zero, _ = joint.forward(model, 'N_level_pattern', cache, arrays,
                            np.array([idx]), rid, rows, details=True)
    checks['zero_gain_vs_bypass'] = float((zero - absent).abs().max())
    assert checks['zero_gain_vs_bypass'] < 1e-4

    reference_rows = [dict(id=str(rows[i]['id']), label=int(rows[i]['label']),
                           category=tr.CLASSES[rows[i]['label']],
                           port=rows[i].get('port'), pass_group=rows[i].get('pass_group')) for i in rid]
    qviews = exp.data.prepare(rows[idx])
    refviews = {k: exp.data.prepare(rows[rid[k]]) for k in range(len(rid))}
    # Select one actual VH/body/same-class relation by its transported content magnitude.
    pol, region, k = 1, 0, same
    op = model.m3[0][pol]
    source = exp.tensor(arrays['x'], rid)[:, pol]
    masks = exp.tensor(arrays['slots'], rid)[:, pol]
    valid = exp.tensor(arrays['valid'], rid)[:, pol]
    masks = torch.minimum(masks.clamp(0, 1), joint.grid_occupancy(valid, source.shape[-2:]).clamp(0, 1))
    mass = exp.tensor(arrays['mass'], rid)[:, pol]
    normalized = F.layer_norm(source.permute(0, 2, 3, 1), (source.shape[1],)).permute(0, 3, 1, 2)
    values = torch.einsum('kshw,kdhw->ksd', masks, op.project(normalized)) / mass.clamp_min(1e-8)[..., None]
    t = tr.arr(details[pol]['transport'][region][0, k])
    w = tr.arr(details[pol]['reference_weights'][0])
    v = tr.arr(values[k, :8])
    qm = tr.arr(exp.tensor(arrays['mass'], [idx])[0, pol, :8])
    offered = qm / max(float(qm.sum()), 1e-20)
    transported = (t @ v) / np.maximum(offered[:, None], 1e-20)
    contribution = w[k, :8, None] * transported
    qi = int(np.linalg.norm(contribution, axis=1).argmax())
    ri = int((t[qi] * np.linalg.norm(v, axis=1)).argmax())
    e = tr.arr(details[pol]['branch_similarity'][region][0, k, qi, ri])[:2]
    alpha = tr.arr(details[pol]['branch_weights'][region][0, k, qi, ri])[:2]
    similarity = tr.arr(details[pol]['similarity'][region][0, k])
    selected = dict(pol='VH', region='框内', reference='31512', query_slot=qi, reference_slot=ri,
                    selection='posthoc same-class31512; maximal norm of weighted transported body row; maximal T*value-norm column',
                    similarities=e.tolist(), weights=alpha.tolist(), terms=(e*alpha).tolist(),
                    similarity=float(similarity[qi, ri]), transport=float(t[qi, ri]),
                    reference_weight=float(w[k, qi]), value_norm=float(np.linalg.norm(v[ri])),
                    row_transport_sum=float(t[qi].sum()), offered=float(offered[qi]))
    receipt = dict(case='12457', truth='散货船', port=rows[idx]['port'], pass_group=rows[idx]['pass_group'],
                   candidate='N_level_pattern', seed=17, m3_frequency=False,
                   scope='Historical cached feature replay. M3 frequency disabled only; not the new current model.',
                   checkpoint=str(joint.HERE / 'training/N_level_pattern_17/m3.pt'),
                   base_checkpoint=str(exp.PACKAGE / 'weights/selected_s17.pt'),
                   conditions=conditions, checks=checks, references=reference_rows,
                   selected=selected, interventions=interventions)
    (ROOT / 'REPLAY_RECEIPT.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame(interventions).drop(columns='logits').to_csv(ROOT / 'local_removal.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(reference_rows).to_csv(ROOT / 'references.csv', index=False, encoding='utf-8-sig')
    np.savez_compressed(ROOT / 'intermediate.npz', selected_values=v, transported=transported,
        weighted_content=contribution, query_feature=tr.arr(original), update=tr.arr(updates),
        **{f'{POL[p]}_{key}_{r}': tr.arr(details[p][key][r]) for p in range(2) for r in range(3)
           for key in ('similarity', 'branch_similarity', 'branch_weights', 'transport')},
        **{f'{POL[p]}_reference_weights': tr.arr(details[p]['reference_weights']) for p in range(2)})

    fig, axes = tr.plt.subplots(2, 3, figsize=(11, 7.6), layout='constrained')
    for p in range(2):
        picture(axes[p, 0], qviews[p], f'{POL[p]} | 12457 查询：散货船', qi if p == 1 else None)
        picture(axes[p, 1], refviews[same][p], f'{POL[p]} | 31512 同类参考', ri if p == 1 else None)
        picture(axes[p, 2], refviews[other][p], f'{POL[p]} | 13000 异类参考：海工')
    fig.suptitle('原生160画布：绿色框内、蓝色侧向、橙色轴端；黑色区域为补边', fontsize=13)
    save(fig, '01_inputs')

    fig, axes = tr.plt.subplots(4, 8, figsize=(19, 10), layout='constrained')
    for k in range(16):
        for p in range(2):
            picture(axes[(k//8)*2+p, k%8], refviews[k][p], f'{ids[k]} {POL[p]}\n{tr.CLASSES[rows[rid[k]]["label"]]}')
    fig.suptitle('固定16个源参考：全部实际输入，8类各2个；标签仅用于说明', fontsize=15)
    save(fig, 'references_native')

    fig, axes = tr.plt.subplots(1, 5, figsize=(18, 4), layout='constrained')
    for c, (matrix, title, cmap, lo, hi) in enumerate([
        (tr.arr(details[1]['branch_similarity'][0][0, same])[..., 0], '深层相似度 E3', 'RdBu_r', -1, 1),
        (tr.arr(details[1]['branch_similarity'][0][0, same])[..., 1], '中层相似度 E2', 'RdBu_r', -1, 1),
        (tr.arr(details[1]['branch_weights'][0][0, same])[..., 0], '深层权重 α3', 'viridis', 0, 1),
        (similarity, '组合相似度 S', 'RdBu_r', -1, 1),
        (t, '软对应 T（非概率）', 'Blues', 0, float(t.max()))]):
        ax = axes[c]; im = ax.imshow(matrix, cmap=cmap, vmin=lo, vmax=hi)
        ax.add_patch(Rectangle((ri-.5, qi-.5), 1, 1, fill=False, ec='#D52F8A', lw=1.7))
        ax.set_title(title, fontsize=11); ax.set_xlabel('参考31512框内槽位')
        ax.set_ylabel('查询12457框内槽位'); fig.colorbar(im, ax=ax, fraction=.045)
    fig.suptitle(f'VH框内：查询槽位{qi} → 同类参考31512槽位{ri}；粉色框为所展示关系', fontsize=14)
    save(fig, '02_compare_match')

    fig, axes = tr.plt.subplots(1, 2, figsize=(12, 5.5), layout='constrained')
    a = picture(axes[0], qviews[1], f'12457 / VH / 查询槽位{qi}', qi)
    b = picture(axes[1], refviews[same][1], f'31512 / VH / 参考槽位{ri}', ri)
    yy, xx = np.mgrid[:160, :160]
    center = lambda m: (float((xx*m).sum()/m.sum()), float((yy*m).sum()/m.sum()))
    link = ConnectionPatch(center(a[qi]), center(b[ri]), 'data', 'data',
                          axesA=axes[0], axesB=axes[1], color='#B2188B', lw=1.5)
    fig.add_artist(link)
    fig.suptitle('实际局部对应回到图像：连线是槽位中心，不是散射点或船首尾标注', fontsize=13)
    save(fig, '03_selected_link')

    fig, axes = tr.plt.subplots(1, 3, figsize=(17, 5.5), layout='constrained')
    for p in range(2):
        weights = tr.arr(details[p]['reference_weights'][0])
        im = axes[p].imshow(weights, aspect='auto', vmin=0, vmax=max(float(d['reference_weights'].max()) for d in details), cmap='viridis')
        axes[p].set_yticks(range(16), ids); axes[p].set_xticks([0,7,8,15,16,19])
        axes[p].axvline(7.5,color='white',lw=.6); axes[p].axvline(15.5,color='white',lw=.6)
        axes[p].set_title(f'{POL[p]}：参考权重 β'); axes[p].set_xlabel('查询槽位'); fig.colorbar(im, ax=axes[p], fraction=.04)
    axes[2].barh(range(16), w[:, qi], color=[tr.GREEN if i == same else tr.BLUE for i in range(16)])
    axes[2].set_yticks(range(16), ids); axes[2].set_title(f'VH查询槽位{qi}的全部参考权重'); axes[2].set_xlabel('β，不是类别概率')
    save(fig, '04_reference_weights')

    fig, axes = tr.plt.subplots(1, 3, figsize=(16, 4.5), layout='constrained')
    lim = max(float(abs(v).max()), float(abs(transported).max()))
    for ax, value, title in zip(axes[:2], [v, transported], ['参考31512的32维槽位内容 V', '按T搬运后的查询槽位内容']):
        im=ax.imshow(value, aspect='auto', cmap='RdBu_r', vmin=-lim, vmax=lim)
        ax.set_title(title); ax.set_xlabel('投影后通道0–31'); ax.set_ylabel('框内槽位0–7'); fig.colorbar(im, ax=ax, fraction=.035)
    axes[2].plot(v[ri], label=f'参考槽位{ri}的V', color=tr.GRAY)
    axes[2].plot(transported[qi], label=f'查询槽位{qi}接收内容（混合多个参考槽位）', color=tr.BLUE)
    axes[2].plot(contribution[qi], label='再乘该参考权重β', color=tr.GREEN)
    axes[2].legend(fontsize=8); axes[2].set_xlabel('投影后通道'); axes[2].set_title('不是复制标签，而是搬运特征向量')
    save(fig, '05_values')

    fig, axes = tr.plt.subplots(1, 2, figsize=(10, 4.5), layout='constrained')
    magnitude = tr.arr(updates[0].norm(dim=1)); maximum = float(magnitude.max())
    for p in range(2):
        im=axes[p].imshow(magnitude[p], cmap='magma', vmin=0, vmax=maximum, interpolation='nearest')
        axes[p].set_title(f'{POL[p]}实际回写 L2范数'); axes[p].set_xlabel('10×10特征网格'); fig.colorbar(im, ax=axes[p], fraction=.04)
    fig.suptitle('通道范数只表示更新强度，不表示对分类必然有益', fontsize=13)
    save(fig, '06_writeback')

    fig, axes = tr.plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    x=np.arange(8)
    for i, record in enumerate(conditions[:2]):
        axes[0].bar(x+(i-.5)*.35, record['logits'], .35, label=record['condition'], color=[tr.GRAY,tr.BLUE][i])
    axes[0].set_xticks(x,tr.CLASSES,rotation=35,ha='right'); axes[0].set_ylabel('分类logit'); axes[0].legend()
    axes[1].barh(range(3), [r['margin'] for r in conditions], color=[tr.GRAY,tr.BLUE,tr.GREEN])
    axes[1].set_yticks(range(3), [r['condition'] for r in conditions]); axes[1].axvline(0,color='black',lw=.8)
    axes[1].set_xlim(-1.15,.5); axes[1].set_xlabel('正确类分差：散货 − 最强其他类')
    for i,r in enumerate(conditions):
        axes[1].text(r['margin']+(.02 if r['margin']>=0 else -.02),i,f'{r["margin"]:+.5f}',va='center',ha='left' if r['margin']<0 else 'left')
    fig.suptitle('真实整模块救回：关闭M3误判化学品/成品油轮，空间M3开启后判为散货',fontsize=13)
    save(fig, '07_prediction')

    fig, ax = tr.plt.subplots(figsize=(11,6), layout='constrained')
    vals=[r['margin'] for r in interventions]
    ax.barh(range(len(vals)), vals, color=[tr.BLUE if v>0 else tr.RED for v in vals])
    ax.set_yticks(range(len(vals)), [r['condition'] for r in interventions]); ax.axvline(0,color='black',lw=.8)
    ax.axvline(conditions[1]['margin'],color=tr.GREEN,ls='--',label='完整空间M3')
    ax.set_xlabel('移除后的正确类分差'); ax.legend(); ax.set_title('固定T、β与剩余内容：单独移除局部回写，不重新分配参考权重')
    save(fig,'08_local_removal')

    # All displayed values are replay outputs; the page is a static research report.
    table=lambda rs, keys: '<table><thead><tr>'+''.join(f'<th>{html.escape(k)}</th>' for k in keys)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{html.escape(str(r[k]))}</td>' for k in keys)+'</tr>' for r in rs)+'</tbody></table>'
    sections=[('1. 原始图像与区域','01_inputs','查询为较长、近竖直的亮主体；31512同类参考斜向，13000海工参考更短且紧凑。这是图像可见事实，不能仅凭这些外观证明散射机制。'),
              ('2. 多层比较与软对应','02_compare_match',f'本图只使用中、深层空间比较。展示VH框内查询槽位{qi}与参考31512槽位{ri}。E3={e[0]:.5f}、E2={e[1]:.5f}；权重分别为{alpha[0]:.5f}、{alpha[1]:.5f}；S={selected["similarity"]:.5f}、T={selected["transport"]:.5f}。该边是事后说明用，未证明单边造成救回。'),
              ('3. 对应落在图像哪里','03_selected_link','先在同类31512的VH框内选加权搬运内容范数最大的查询行，再选T乘value范数最大的列；不是挑选已证实因果最大的像素。匹配连线连接槽位重心。'),
              ('4. 16个参考如何聚合','04_reference_weights',f'该查询槽位分给31512的权重β={selected["reference_weight"]:.5f}。所有16个参考均参与合法性判断；标签不进入推理期匹配。高权重不等于有益内容。'),
              ('5. 通道内容如何传递','05_values','参考特征先按通道归一化，经1×1投影从1024降至32维；T搬运并按查询槽位质量归一化，再乘β。热图显示真实带符号特征，不是原始SAR强度。'),
              ('6. 回写到查询特征','06_writeback','槽位内容按查询掩膜回写；1×1投影恢复1024维，乘增益后与查询残差相加。随后layer4和多层区域读出重新计算。本图为整个空间M3回写，不是频域开关差异。'),
              ('7. 最终分类翻转','07_prediction','同一权重、同一查询、同一16参考，仅关闭全部M3回写：化学品/成品油轮→散货船。关闭方式另用gain=0重放核对，结果相同。这是固定模型路径依赖，不是独立重训的无M3对照。'),
              ('8. 哪些回写实际支撑判别','08_local_removal','只移除指定回写项，不修改原图，不重新分配T或β。局部移除效果经过后续非线性网络，不能直接相加；同类参考也不自动比异类参考更有用。')]
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>12457：空间M3真实救回全流程</title><style>
body{font:16px/1.8 system-ui,"Microsoft YaHei",sans-serif;color:#202a30;background:#fff;margin:0}main{max-width:1280px;margin:auto;padding:28px 24px}h1{font-size:28px;line-height:1.4}h2{font-size:22px;border-top:1px solid #d7dce0;padding-top:24px;margin-top:32px}img{width:100%;height:auto}a{color:#2166ac}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}figure{margin:20px 0}figcaption{font-size:14px;color:#56616b}.flow{display:flex;gap:12px;overflow-x:auto;padding:16px 0}.step{flex:0 0 170px;border-top:3px solid #2166ac;padding:12px;background:#f5f8fa}.note{border-left:4px solid #1b7837;padding:8px 16px;background:#f3f8f4}.scroll{overflow:auto}@media(max-width:600px){main{padding:18px 12px}h1{font-size:24px}}
</style><main><h1>12457：空间M3如何救回一艘散货船</h1>
<p>Jebel Ali · S1A_058688_074483 · 历史N_level_pattern / seed17 · 固定16个源参考 · 不重训</p>
<p class="note"><b>确认真实整模块救回：</b>关闭M3回写分差−0.83598，误判化学品/成品油轮；开启空间M3后分差+0.13756，判为散货船。主图关闭M3频域比较；这不是此前24470的“仅频域开关救回”。</p>
<p>边界：使用历史候选及其固定上游缓存，保留历史M1/M2，不是2026-10-31新整合模型的准确率证据。没有修改正式代码或检查点；一个案例不代表总体收益。</p>
<div class="flow">'''+''.join(f'<div class="step"><b>{a}</b><br>{b}</div>' for a,b in [('① 区域读出','20个槽位；VV/VH分开'),('② 空间比较','中层＋深层 → S'),('③ 软匹配','区域内T＋参考权重β'),('④ 内容搬运','32维V → T对齐 → β聚合'),('⑤ 特征回写','回到10×10网格，残差相加'),('⑥ 分类','误判化学品船 → 散货船')])+ '</div>'
    page += '<div class="scroll">'+table(conditions,['condition','predicted_class','margin'])+'</div><h2>横向总图：真实数据贯穿整个M3</h2><a href="09_overview.pdf"><img src="09_overview.png" alt="12457真实空间M3全流程"></a><p>总图只选VH框内的一条关系示范计算，实际输出由两极化、三区域和全部合法参考共同形成；下方逐步展开。</p>'
    for heading,name,body in sections:
        page+=f'<section><h2>{heading}</h2><p>{body}</p><figure><a href="{name}.pdf"><img src="{name}.png" alt="{heading}" loading="lazy"></a><figcaption><a href="{name}.pdf">矢量PDF</a> · <a href="{name}.png">原尺寸PNG</a></figcaption></figure></section>'
    page += '<h2>固定参考与可复现记录</h2><img src="references_native.png" alt="16个源参考原图" loading="lazy"><div class="scroll">'+table(reference_rows,['id','category','port','pass_group'])+'</div><p><a href="REPLAY_RECEIPT.json">重放数值与核验</a> · <a href="local_removal.csv">局部关闭结果</a> · <a href="references.csv">固定参考列表</a></p></main></html>'
    (ROOT/'index.html').write_text(page,encoding='utf-8')
    import overview
    overview.draw()
    print(json.dumps(dict(conditions=conditions,checks=checks,selected=selected,interventions=interventions),ensure_ascii=True),flush=True)


if __name__ == '__main__':
    main()
