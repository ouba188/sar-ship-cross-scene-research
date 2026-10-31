"""Paired summaries, aggregation diagnostics and native-chip case review."""
import json
import html
import numpy as np
import pandas as pd
import torch
import run_compare as rc

ROOT, tr, exp, joint = rc.ROOT, rc.tr, rc.exp, rc.joint
LABEL = dict(soft='精细软匹配', pool_keep_beta_mass='区域平均：保留原权重/质量',
             uniform_refs='局部匹配：参考均匀', pool_uniform_mass='区域平均：参考均匀/保留质量',
             simple_region_mean='简单区域均值聚合', m3_off='关闭M3回写')


def save(fig, name):
    fig.savefig(ROOT/f'{name}.png', dpi=150, bbox_inches='tight', facecolor='white')
    tr.plt.close(fig)


def picture(ax, view, title):
    if view is None:
        ax.text(.5, .5, '极化缺失', ha='center', va='center', transform=ax.transAxes)
    else:
        ax.imshow(view['pixels'], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        masks = exp.tokens.slots(view['geometry'], view['footprint'])
        for indexes, color in zip(exp.tokens.REGIONS, ('#008570', '#226FB6', '#B76820')):
            mask = masks[indexes].sum(0)
            if mask.max() > .05:
                ax.contour(mask, levels=[.05], colors=[color], linewidths=.6)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


@torch.no_grad()
def main():
    predictions = pd.read_csv(ROOT/'predictions.csv', dtype={'id': str})
    metrics = pd.read_csv(ROOT/'metrics.csv')
    summary = metrics.groupby('arm')[['oa', 'ba', 'nll', 'brier']].agg(['mean', 'std']).reindex(rc.ARMS)
    summary.to_csv(ROOT/'summary.csv')
    paired = []
    for seed, part in predictions.groupby('seed'):
        w = part.pivot(index='id', columns='arm', values='prediction')
        y = part.drop_duplicates('id').set_index('id').label.reindex(w.index)
        for comparator in ('soft', 'm3_off'):
            for arm in rc.ARMS:
                paired.append(dict(seed=int(seed), arm=arm, comparator=comparator,
                    changed=int((w[arm] != w[comparator]).sum()),
                    rescued=int(((w[arm] == y) & (w[comparator] != y)).sum()),
                    harmed=int(((w[arm] != y) & (w[comparator] == y)).sum())))
    pd.DataFrame(paired).to_csv(ROOT/'paired_counts.csv', index=False)
    class_rows = []
    for (seed, arm, label), group in predictions.groupby(['seed', 'arm', 'label']):
        class_rows.append(dict(seed=seed, arm=arm, label=label, category=tr.CLASSES[label], n=len(group), recall=100*(group.prediction==label).mean()))
    pd.DataFrame(class_rows).to_csv(ROOT/'class_recalls.csv', index=False)
    fig, axes = tr.plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric, name in zip(axes, ('oa', 'ba'), ('OA (%)', 'BA (%)')):
        for i, arm in enumerate(rc.ARMS):
            v = metrics[metrics.arm.eq(arm)][metric].to_numpy()
            ax.scatter(np.repeat(i, len(v)), v, s=24, color='#226FB6')
            ax.plot([i-.18, i+.18], [v.mean()]*2, color='#008570', linewidth=2)
        ax.set_xticks(range(6), ['软匹配', '区域平均\n原权重/质量', '匹配\n均匀参考', '区域平均\n均匀/原质量', '简单\n区域均值', '关闭M3'], fontsize=8)
        ax.set_ylabel(name); ax.grid(axis='y', alpha=.2)
    fig.suptitle('160张开发查询，固定16参考；三个历史M3训练种子，推理替换而非重训', fontsize=11)
    fig.tight_layout(); save(fig, 'metrics')

    exp.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    exp.current.setup(exp.DEVICE); torch.set_num_threads(2)
    rows, nf, refs, rid, arrays = exp.load()
    cache = joint.pair_cache()
    diagnostics = []
    update_diffs = []
    for seed in (17, 29, 43):
        model = tr.review.restore('N_level_pattern', seed)
        model.requires_grad_(False)
        for op in model.m3[0]:
            op.frequency = False
        rv = [rc.reference_values(model, arrays, rid, p) for p in range(2)]
        for start in range(nf, len(rows), 16):
            ids = np.arange(start, min(start+16, len(rows)))
            _, ds = joint.forward(model, 'N_level_pattern', cache, arrays, ids, rid, rows, details=True)
            qm, slots, valid = [exp.tensor(arrays[k], ids) for k in ('mass', 'slots', 'valid')]
            avail, ravail = exp.tensor(arrays['available'], ids).bool(), exp.tensor(arrays['available'], rid).bool()
            legal = exp.data.pair_mask([rows[i] for i in ids], [rows[i] for i in rid], exp.DEVICE) & model.m3[0][0].cross_orbit
            for p in range(2):
                for r, ix in enumerate(exp.tokens.REGIONS):
                    t = ds[p]['transport'][r]
                    mass = t.sum(-1)
                    conditional = t/mass[..., None].clamp_min(1e-20)
                    rm = rv[p][1][None, :, ix]
                    area = rm/rm.sum(-1, keepdim=True).clamp_min(1e-20)
                    tv = .5*(conditional-area[..., None, :]).abs().sum(-1)
                    beta = ds[p]['reference_weights'][:, :, ix]
                    live = mass > 0
                    uniform = live.to(beta)/live.sum(1, keepdim=True).clamp_min(1)
                    beta_tv = .5*(beta-uniform).abs().sum(1)
                    value = rv[p][0][:, ix]
                    mean = (value*area[0, :, :, None]).sum(1)
                    dispersion = ((value-mean[:, None]).square().sum(-1)*area[0]).sum(-1).sqrt()/mean.norm(dim=-1).clamp_min(1e-8)
                    for j, idx in enumerate(ids):
                        active = live[j]
                        diagnostics.append(dict(seed=seed, id=str(rows[idx]['id']), pol=tr.POL[p], region=tr.REGIONS[r],
                            local_tv=float(tv[j][active].mean()) if active.any() else np.nan,
                            reference_tv=float(beta_tv[j][active.any(0)].mean()) if active.any() else np.nan,
                            value_dispersion=float(dispersion[active.any(-1)].mean()) if active.any() else np.nan))
                for arm in ('pool_keep_beta_mass', 'simple_region_mean'):
                    update = rc.replace_update(model.m3[0][p], ds[p], *rv[p], qm[:, p], slots[:, p], valid[:, p],
                        legal & avail[:, p, None] & ravail[None, :, p], arm)
                    difference = (update-ds[p]['update']).flatten(1).norm(dim=-1)/ds[p]['update'].flatten(1).norm(dim=-1).clamp_min(1e-8)
                    for j, idx in enumerate(ids):
                        update_diffs.append(dict(seed=seed, id=str(rows[idx]['id']), pol=tr.POL[p], arm=arm, relative_update_difference=float(difference[j])))
        del model
    pd.DataFrame(diagnostics).to_csv(ROOT/'mechanism_diagnostics.csv', index=False)
    pd.DataFrame(update_diffs).to_csv(ROOT/'update_differences.csv', index=False)
    print('DIAGNOSTICS', pd.DataFrame(diagnostics).groupby('pol')[['local_tv','reference_tv','value_dispersion']].mean().to_string(), flush=True)
    print('UPDATE_DIFFERENCE', pd.DataFrame(update_diffs).groupby(['arm','pol']).relative_update_difference.agg(['mean','max']).to_string(), flush=True)

    s = predictions[predictions.seed.eq(17)]
    w = s.pivot(index='id', columns='arm', values='prediction')
    truth = s.drop_duplicates('id').set_index('id').label.reindex(w.index)
    stable_correct = w.index[(w.soft==truth)&(w.m3_off==truth)][0]
    stable_wrong = w.index[(w.soft!=truth)&(w.m3_off!=truth)][0]
    chosen = [('12457', '被救回'), ('24470', '被误伤'), (stable_correct, '持续判对'), (stable_wrong, '持续判错')]
    flipped = predictions[predictions.seed.eq(43)].pivot(index='id', columns='arm', values='prediction')
    extra = flipped.index[flipped.soft != flipped.uniform_refs].tolist()
    if extra:
        chosen.append((extra[0], '种子43均匀参考救回'))
    cases = []
    for case, kind in chosen:
        idx = next(i for i in range(nf, len(rows)) if str(rows[i]['id']) == case)
        row = rows[idx]
        seed = 43 if kind.startswith('种子43') else 17
        selected = predictions[(predictions.seed==seed)&(predictions.id==case)]
        pred = selected.set_index('arm').prediction
        same = next(i for i in rid if rows[i]['label']==row['label'])
        confused_class = int(pred['m3_off'] if pred['m3_off']!=row['label'] else pred['soft'])
        if confused_class==row['label']:
            score = np.load(ROOT/f'{seed}_soft_logits.npy')[idx-nf].copy(); score[row['label']] = -np.inf
            confused_class = int(score.argmax())
        other = next(i for i in rid if rows[i]['label']==confused_class)
        # Fixed first actual reference of the relevant class, not a hand-picked best-looking match.
        fig, axes = tr.plt.subplots(2, 3, figsize=(11, 9), layout='constrained')
        for col, k in enumerate((idx, same, other)):
            views = exp.data.prepare(rows[k])
            for p in range(2):
                picture(axes[p, col], views[p], f'{("查询", "同类参考", "混淆类参考")[col]} {rows[k]["id"]} / {tr.POL[p]}\n{tr.CLASSES[rows[k]["label"]]}')
        margins = selected.set_index('arm').margin
        fig.suptitle(f'{kind} {case} / seed {seed} / {row.get("port")}\n关闭M3 {margins["m3_off"]:+.3f} → 软匹配 {margins["soft"]:+.3f} / 简单均值 {margins["simple_region_mean"]:+.3f}', fontsize=12)
        save(fig, f'case_{case}_s{seed}')
        cases.append(dict(id=case, kind=kind, seed=seed, port=row.get('port'), pass_group=row.get('pass_group'), label=row['label'],
                          same_reference=str(rows[same]['id']), confusion_reference=str(rows[other]['id']),
                          source_row=row, predictions=selected.to_dict('records')))
    (ROOT/'cases.json').write_text(json.dumps(cases, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    parts = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>M3软匹配与简单聚合</title>',
             '<style>body{font:16px/1.65 sans-serif;max-width:1150px;margin:30px auto;padding:0 20px;color:#172323}img{max-width:100%}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #bbb}</style>',
             '<h1>M3软匹配与简单参考聚合：冻结替换对照</h1>',
             '<p>160张开发查询、固定16源参考、三个历史M3种子。未重训、未改正式模型。M3显式频域关闭；历史模型其他部分不在本轮修改范围。</p>',
             '<p><b>精细局部匹配换成区域均值，480次配对预测均未改变。简单区域均值也保留12457的救回。</b>这不证明分别训练后的架构等价。</p>',
             '<img src="metrics.png">', metrics.round(6).to_html(index=False),
             '<p><a href="FINDINGS_ZH.md">完整结论与原图观察</a> / <a href="paired_counts.csv">逐种子救回与误伤</a> / <a href="mechanism_diagnostics.csv">机制量化</a></p>']
    for c in cases:
        parts += [f'<h2>{html.escape(c["kind"])}：{c["id"]}</h2><img src="case_{c["id"]}_s{c["seed"]}.png">',
                  pd.DataFrame(c['predictions'])[['arm','prediction','margin']].round(6).to_html(index=False)]
    parts += ['<p>图像使用实际输入视图与原有显示标度，无额外拉伸。绿色框内，蓝色侧向，橙色轴端；不据外观命名物理散射机制。</p></html>']
    (ROOT/'index.html').write_text('\n'.join(parts), encoding='utf-8')
    print('SUMMARY', summary.to_string(), flush=True)
    print('CASES', [(c['id'],c['kind']) for c in cases], flush=True)


if __name__ == '__main__':
    main()
