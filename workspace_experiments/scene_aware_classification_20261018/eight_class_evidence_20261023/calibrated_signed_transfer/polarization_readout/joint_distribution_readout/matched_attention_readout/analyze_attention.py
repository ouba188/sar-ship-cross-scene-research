"""Paired fixed-prediction analysis and native panels, without additional fitting."""
from common import *
import argparse
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def per_sample_risk(y, score):
    shifted = score-score.max(1, keepdims=True)
    logp = shifted-np.log(np.exp(shifted).sum(1, keepdims=True))
    prob = np.exp(logp)
    return -logp[np.arange(len(y)), y], ((prob-np.eye(8)[y])**2).sum(1)


def intervals(g, before, label):
    y = g.class_id.to_numpy()
    after = g[SCORE].to_numpy()
    diff = (after.argmax(1) == y).astype(int)-(before.argmax(1) == y).astype(int)
    an, ab = per_sample_risk(y, after)
    bn, bb = per_sample_risk(y, before)
    results = []
    for block in ['pass_group', 'mmsi']:
        codes, names = pd.factorize(g[block], sort=True)
        count = np.zeros((len(names), 8))
        gain = np.zeros_like(count)
        risk = np.zeros((len(names), 2))
        np.add.at(count, (codes, y), 1)
        np.add.at(gain, (codes, y), diff)
        np.add.at(risk, codes, np.stack((an-bn, ab-bb), 1))
        rng = np.random.default_rng(20261023)
        samples = []
        for _ in range(2000):
            idx = rng.integers(0, len(names), len(names))
            den, num = count[idx].sum(0), gain[idx].sum(0)
            r = risk[idx].sum(0)/den.sum()
            samples.append([100*num.sum()/den.sum(),
                100*np.mean(num/den) if (den > 0).all() else np.nan, *r])
        samples = np.array(samples)
        for c, metric in enumerate(['OA', 'BA', 'NLL', 'Brier']):
            vals = samples[:, c]
            results.append(dict(reference=label, block=block, blocks=len(names), metric=metric,
                draws=2000, finite_draws=int(np.isfinite(vals).sum()),
                low=float(np.nanquantile(vals, .025)), high=float(np.nanquantile(vals, .975))))
    return results


def render(p, stage):
    table = parent.load_table().set_index('instance_id')
    cache = cache_load()
    ac = dict(np.load(HERE/f'attention_{stage}.npz'))
    lookup = {s: i for i, s in enumerate(cache['instance_ids'])}
    snapshots = {s['obs_id']: s for s in json.loads((run.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    role = 'validation' if stage == 'source' else 'test'
    target = p[p.arm.eq(ARM) & p.seed.eq(17) & p.role.eq(role)]
    full = p[p.arm.eq('full') & p.seed.eq(17) & p.role.eq(role)].set_index(['fold', 'instance_id'])
    (HERE/'panels').mkdir(exist_ok=True)
    cases, missing = [], []
    for outcome in ['rescued', 'harmed', 'CC', 'WW']:
        hits = target[target.outcome.eq(outcome)].sort_values(['instance_id', 'fold'])
        if not len(hits):
            missing.append(outcome)
            continue
        q = hits.iloc[0]
        row = table.loc[q.instance_id]
        y, rival = int(q.class_id), int(q.final_rival)
        b, s = q[BASE].to_numpy(float), q[SCORE].to_numpy(float)
        fs = full.loc[(q.fold, q.instance_id), SCORE].to_numpy(float)
        ri = list(cache['reader_names']).index(f'fold{int(q.fold)}_outer')
        i = lookup[q.instance_id]
        hb, sk = ac['attention_pol'][ri, i], cache['moments_pol'][ri, i]
        fig, axes = plt.subplots(2, 3, figsize=(15, 10), layout='constrained')
        shapes = []
        for pol, label in enumerate(['vv', 'vh']):
            view = snapshots[int(row.obs_id)]['images'][label]
            with Image.open(row[f'image_{label}']) as im:
                raw = np.array(im.convert('L'))
            pix, masks, place = extract.crop_pad(raw, extract.masks_for(view), 160)
            shapes.append(f'{label}:{raw.shape[1]}x{raw.shape[0]}')
            axes[pol, 0].imshow(raw, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
            axes[pol, 0].set_title(f'{label.upper()} original {raw.shape[1]}x{raw.shape[0]} pixels')
            axes[pol, 1].imshow(pix, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
            axes[pol, 1].set_title(f'Actual 160 crop/pad; valid={q[f"valid_{label}"]}\nfootprint={masks.sum(0).mean():.3f}')
            prior = view['priors']
            target_index, detections = prior.get('target_index'), prior.get('detections', [])
            if target_index is not None and 0 <= target_index < len(detections):
                corners = np.array(detections[target_index]['corners'])
                points = np.r_[corners, corners[:1]]
                axes[pol, 0].plot(points[:, 0], points[:, 1], color='#13c74e', lw=1)
                points = points-[place['crop_x'], place['crop_y']]+[place['pad_left'], place['pad_top']]
                axes[pol, 1].plot(points[:, 0], points[:, 1], color='#13c74e', lw=1)
            for ax in axes[pol, :2]:
                ax.set_xlabel('pixel x')
                ax.set_ylabel('pixel y')
        xx = np.arange(8)
        for offset, values, label, color in [(-.24, b, 'Frozen B = T(oldAD)', '#777777'),
            (0, fs, 'Full moments', '#1463a3'), (.24, s, 'Two-sided attention', '#168b65')]:
            axes[0, 2].bar(xx+offset, values, .24, label=label, color=color)
        axes[0, 2].set_xticks(xx)
        axes[0, 2].set_title('Actual class scores, same frozen B')
        axes[0, 2].legend(fontsize=8)
        values = hb[:, [y, rival], :].transpose(0, 2, 1).reshape(4, 2)
        axes[1, 2].bar(np.arange(4)-.17, values[:, 0], .34, label=f'true class {y}', color='#168b65')
        axes[1, 2].bar(np.arange(4)+.17, values[:, 1], .34, label=f'fixed rival {rival}', color='#b65169')
        axes[1, 2].set_xticks(np.arange(4), ['VV h', 'VV b', 'VH h', 'VH b'])
        axes[1, 2].axhline(0, color='black', lw=.6)
        axes[1, 2].legend(fontsize=8)
        axes[1, 2].set_title('C4-averaged INPUT summaries\nNot spatial/physical attribution')
        fig.suptitle(f'{stage} / seed17 / fold{int(q.fold)} / epoch{int(q.epoch)} / {outcome} versus B\n'
            f'{q.instance_id} / {row.class_name} / {row.port}\n'
            f'true={y}; B={int(q.base_prediction)}, full={int(fs.argmax())}, attention={int(q.prediction)}; '
            f'old rival={int(q.old_rival)}, attention rival={rival}\n'
            f'Fixed rival margin B {b[y]-b[rival]:+.5f}, full {fs[y]-fs[rival]:+.5f}, '
            f'attention {s[y]-s[rival]:+.5f}; attention increment {(s-b)[y]-(s-b)[rival]:+.5f}', fontsize=11)
        path = HERE/'panels'/f'{stage}_{outcome}.png'
        fig.savefig(path, dpi=105)
        plt.close(fig)
        cases.append(dict(stage=stage, role=role, seed=17, fold=int(q.fold), epoch=int(q.epoch),
            outcome=outcome, instance_id=q.instance_id, class_id=y, class_name=row.class_name, port=row.port,
            base_prediction=int(q.base_prediction), full_prediction=int(fs.argmax()), prediction=int(q.prediction),
            old_rival=int(q.old_rival), final_rival=rival, base_margin=float(b[y]-b[rival]),
            full_margin=float(fs[y]-fs[rival]), final_margin=float(s[y]-s[rival]),
            image_vv=row.image_vv, image_vh=row.image_vh, dimensions=';'.join(shapes),
            valid_vh=bool(q.valid_vh), panel=str(path), selection='seed17 first ID then fold per outcome versus B',
            sigma_true=sk[:, y, 0].tolist(), kappa_true=sk[:, y, 1].tolist(),
            h_true=hb[:, y, 0].tolist(), b_true=hb[:, y, 1].tolist()))
    pd.DataFrame(cases).to_csv(HERE/f'{stage}_review_cases.csv', index=False)
    return cases, missing


def main(stage):
    event('analysis_start', stage=stage)
    p = pd.read_csv(HERE/f'{stage}_predictions.csv', dtype={'mmsi': str}, float_precision='round_trip')
    role = 'validation' if stage == 'source' else 'test'
    pairs, classes, strata, ci = [], [], [], []
    for (current_role, seed), a in p[p.arm.eq(ARM)].groupby(['role', 'seed']):
        a = a.set_index(['fold', 'instance_id']).sort_index()
        f = p[p.arm.eq('full') & p.seed.eq(seed) & p.role.eq(current_role)].set_index(['fold', 'instance_id']).loc[a.index]
        np.testing.assert_array_equal(a.class_id, f.class_id)
        np.testing.assert_array_equal(a[BASE], f[BASE])
        y, scores = a.class_id.to_numpy(), a[SCORE].to_numpy()
        am = parent.assessment(y, scores)
        for reference, before in [('B', a[BASE].to_numpy()), ('full', f[SCORE].to_numpy())]:
            rm = parent.assessment(y, before)
            diff = (scores.argmax(1) == y).astype(int)-(before.argmax(1) == y).astype(int)
            pairs.append(dict(role=current_role, seed=int(seed), reference=reference, n=len(a), net=int(diff.sum()),
                rescued=int((diff == 1).sum()), harmed=int((diff == -1).sum()),
                **{f'delta_{k}': am[k]-rm[k] for k in ['OA', 'BA', 'J', 'NLL', 'Brier']}))
            for cls in range(8):
                take = y == cls
                classes.append(dict(role=current_role, seed=int(seed), reference=reference, class_id=cls,
                    n=int(take.sum()), net=int(diff[take].sum()), delta_recall=100*float(diff[take].mean()),
                    rescued=int((diff[take] == 1).sum()), harmed=int((diff[take] == -1).sum())))
            if current_role == 'test':
                ci.extend(dict(seed=int(seed), **x) for x in intervals(a, before, reference))
            for unit in ['fold', 'port', 'pass_group', 'product', 'valid_vh']:
                values = a.index.get_level_values('fold').to_numpy() if unit == 'fold' else a[unit].to_numpy()
                for value in pd.unique(values):
                    take = values == value
                    nm, bm = parent.assessment(y[take], scores[take]), parent.assessment(y[take], before[take])
                    strata.append(dict(role=current_role, seed=int(seed), reference=reference, unit=unit, name=value,
                        n=int(take.sum()), net=int(diff[take].sum()), **{f'delta_{k}': nm[k]-bm[k] for k in ['OA','BA','NLL','Brier']}))
    for name, rows in [('paired_metrics', pairs), ('paired_classes', classes), ('stratified', strata), ('intervals', ci)]:
        pd.DataFrame(rows).to_csv(HERE/f'{stage}_{name}.csv', index=False)
    cases, missing = render(p, stage)
    write_json(f'{stage}_analysis.json', dict(completed_utc=now(), panels=[x['panel'] for x in cases],
        missing_outcomes=missing, raw_images_viewed=False, new_fits=0,
        ci_scope='Fixed predictions; pass/MMSI blocks, excludes selection/fitting uncertainty'))
    source_or_target = '源验证' if stage == 'source' else '历史test'
    lines = [f'# {source_or_target}：固定two-sided attention替换', '',
        '本文件先保存可复算数值；原图实际查看记录另列IMAGE_NOTES。无新增臂/温度搜索。', '',
        '|seed|参照|净数|ΔOA pp|ΔBA pp|ΔNLL|ΔBrier|', '|---|---|---:|---:|---:|---:|---:|']
    for r in pairs:
        if r['role'] == role:
            lines.append(f"|{r['seed']}|{r['reference']}|{r['net']:+d}|{r['delta_OA']:+.4f}|{r['delta_BA']:+.4f}|{r['delta_NLL']:+.6f}|{r['delta_Brier']:+.6f}|")
    lines.extend(['', '差值均为attention减参照；风险指标越低越好。full不是重新训练。',
        '源验证参与选点，不是独立确认；T非二级OOF。历史test属于已开发队列，不按其结果改模型。',
        '预测、逐类及采集表、选择曲线、实际提取/训练成本、时间顺序分别保存在本目录。',
        '缺失结局：'+str(missing)+ '。'])
    (HERE/('SOURCE_FINDINGS_ZH.md' if stage == 'source' else 'TEST_NUMERICAL_ZH.md')).write_text('\n'.join(lines)+'\n', encoding='utf-8')
    event('analysis_tables_complete', stage=stage)
    print(pd.DataFrame(pairs).query('role==@role').to_string(index=False), flush=True)
    print(pd.DataFrame(cases)[['outcome', 'instance_id', 'class_name', 'base_prediction', 'full_prediction', 'prediction', 'base_margin', 'full_margin', 'final_margin']].to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['source', 'test'])
    main(parser.parse_args().stage)
