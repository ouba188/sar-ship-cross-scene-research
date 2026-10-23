"""Fixed source/test comparisons and bounded raw case review for side restrictions."""
from run_sides import *
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def render(p):
    table = u.parent.load_table().set_index('instance_id')
    cache = u.cache_load()
    ac = original.load_attention('test', cache)
    lookup = {s: i for i, s in enumerate(cache['instance_ids'])}
    snaps = {s['obs_id']: s for s in json.loads((u.run.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    g = p[p.seed.eq(17)]
    byarm = {arm: sub.set_index(['fold', 'instance_id']).sort_index() for arm, sub in g.groupby('arm')}
    high, low = byarm['high'], byarm['low']
    low = low.loc[high.index]
    hc, lc = high.prediction.eq(high.class_id), low.prediction.eq(low.class_id)
    fixed = ['pair_0005030', 'pair_0007310', 'pair_0003494', 'pair_0003504']
    choices = [(int(high[high.index.get_level_values('instance_id') == ident].index[0][0]), ident, 'fixed') for ident in fixed]
    fixed_mask = high.index.get_level_values('instance_id').isin(fixed)
    missing = []
    for tag, take in [('high_only_correct', hc & ~lc), ('low_only_correct', lc & ~hc)]:
        if (take & fixed_mask).any():
            continue
        hits = high[take].reset_index().sort_values(['instance_id', 'fold'])
        if len(hits):
            q = hits.iloc[0]
            choices.append((int(q.fold), q.instance_id, tag))
        else:
            missing.append(tag)
    (HERE/'panels').mkdir(exist_ok=True)
    cases = []
    for fold, ident, reason in choices:
        row = table.loc[ident]
        preds = {arm: data.loc[(fold, ident)] for arm, data in byarm.items()}
        y = int(row.class_id)
        b = preds['two_sided'][BASE].to_numpy(float)
        scores = {arm: q[SCORE].to_numpy(float) for arm, q in preds.items()}
        rival = int(preds['two_sided'].final_rival)
        ri = list(cache['reader_names']).index(f'fold{fold}_outer')
        hb = ac['attention_pol'][ri, lookup[ident]]
        qp, qm = hb[..., 0]+hb[..., 1], hb[..., 1]-hb[..., 0]
        fig, axes = plt.subplots(2, 3, figsize=(16, 10), layout='constrained')
        dims = []
        for pol, label in enumerate(['vv', 'vh']):
            view = snaps[int(row.obs_id)]['images'][label]
            with Image.open(row[f'image_{label}']) as im:
                raw = np.array(im.convert('L'))
            pix, mask, place = u.extract.crop_pad(raw, u.extract.masks_for(view), 160)
            dims.append(f'{label}:{raw.shape[1]}x{raw.shape[0]}')
            axes[pol, 0].imshow(raw, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
            axes[pol, 0].set_title(f'{label.upper()} original {raw.shape[1]}x{raw.shape[0]} pixels')
            axes[pol, 1].imshow(pix, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
            axes[pol, 1].set_title(f'Actual 160 crop/pad; valid={preds["high"][f"valid_{label}"]}')
            prior = view['priors']
            target, detections = prior.get('target_index'), prior.get('detections', [])
            if target is not None and 0 <= target < len(detections):
                points = np.array(detections[target]['corners'])
                points = np.r_[points, points[:1]]
                axes[pol, 0].plot(points[:, 0], points[:, 1], color='#13c74e', lw=1)
                points = points-[place['crop_x'], place['crop_y']]+[place['pad_left'], place['pad_top']]
                axes[pol, 1].plot(points[:, 0], points[:, 1], color='#13c74e', lw=1)
            for ax in axes[pol, :2]:
                ax.set_xlabel('pixel x')
                ax.set_ylabel('pixel y')
        xs = np.arange(8)
        all_scores = {'B': b, **{k: scores[k] for k in ['two_sided', 'high', 'low']}}
        for offset, (arm, score), color in zip([-.27, -.09, .09, .27], all_scores.items(), ['#777777','#1463a3','#b25a27','#168b65']):
            axes[0, 2].bar(xs+offset, score, .18, label=arm, color=color)
        axes[0, 2].set_xticks(xs)
        axes[0, 2].legend(fontsize=8)
        axes[0, 2].set_title('Actual scores; B frozen calibration')
        values = np.array([[qp[0,y], qp[0,rival]], [qm[0,y], qm[0,rival]], [qp[1,y], qp[1,rival]], [qm[1,y], qm[1,rival]]])
        axes[1, 2].bar(np.arange(4)-.17, values[:, 0], .34, label=f'true {y}', color='#168b65')
        axes[1, 2].bar(np.arange(4)+.17, values[:, 1], .34, label=f'fixed rival {rival}', color='#b65169')
        axes[1, 2].set_xticks(np.arange(4), ['VV q+', 'VV q-', 'VH q+', 'VH q-'])
        axes[1, 2].axhline(0, color='black', lw=.7)
        axes[1, 2].legend(fontsize=8)
        axes[1, 2].set_title('Input summaries, not output attribution\nhigh uses only q+; low only q-')
        winners = '; '.join(f'{arm}={int(s.argmax())}' for arm, s in all_scores.items())
        epochs = '; '.join(f'{arm}:{int(preds[arm].epoch)}' for arm in ['two_sided','high','low'])
        margins = '; '.join(f'{arm}:{s[y]-s[rival]:+.5f}' for arm, s in all_scores.items())
        fig.suptitle(f'{ident} / {row.class_name} / {row.port} / seed17 fold{fold} / {reason}\n'
            f'true={y}; {winners}\nSelected epochs {epochs}\nFixed two-sided rival {rival} margins: {margins}', fontsize=10)
        name = ident.replace('pair_', '') if reason == 'fixed' else reason
        path = HERE/'panels'/f'{name}.png'
        fig.savefig(path, dpi=105)
        plt.close(fig)
        result = dict(instance_id=ident, seed=17, fold=fold, reason=reason, class_id=y, class_name=row.class_name,
            port=row.port, rival=rival, base_prediction=int(b.argmax()), base_margin=float(b[y]-b[rival]),
            image_vv=row.image_vv, image_vh=row.image_vh, dimensions=';'.join(dims), panel=str(path))
        for arm in ['two_sided', 'high', 'low']:
            result.update({f'{arm}_prediction': int(scores[arm].argmax()), f'{arm}_epoch': int(preds[arm].epoch),
                f'{arm}_margin': float(scores[arm][y]-scores[arm][rival]),
                f'{arm}_outcome_vs_B': preds[arm].outcome})
        cases.append(result)
    pd.DataFrame(cases).to_csv(HERE/'review_cases.csv', index=False)
    save_json('image_review_status.json', dict(panels=len(cases), raw_images_viewed=False, missing_extra_outcomes=missing))
    return cases


def analyze(stage):
    event('analysis_start', stage=stage)
    p = pd.read_csv(HERE/f'{stage}_predictions.csv', dtype={'mmsi': str}, float_precision='round_trip')
    comparisons = [('high','B'), ('low','B'), ('two_sided','B'), ('two_sided','high'), ('two_sided','low'), ('high','low')]
    pairs, classes, strata = [], [], []
    for (role, seed), g in p.groupby(['role', 'seed']):
        arms = {a: sub.set_index(['fold', 'instance_id']).sort_index() for a, sub in g.groupby('arm')}
        for arm, reference in comparisons:
            a = arms[arm]
            y, score = a.class_id.to_numpy(), a[SCORE].to_numpy()
            if reference == 'B':
                before = a[BASE].to_numpy()
            else:
                ref = arms[reference].loc[a.index]
                np.testing.assert_array_equal(ref.class_id, a.class_id)
                np.testing.assert_array_equal(ref[BASE], a[BASE])
                before = ref[SCORE].to_numpy()
            am, bm = u.parent.assessment(y, score), u.parent.assessment(y, before)
            diff = (score.argmax(1)==y).astype(int)-(before.argmax(1)==y).astype(int)
            meta = dict(role=role, seed=int(seed), arm=arm, reference=reference)
            pairs.append(dict(**meta, n=len(a), net=int(diff.sum()), rescued=int((diff==1).sum()), harmed=int((diff==-1).sum()),
                **{f'delta_{k}': am[k]-bm[k] for k in ['OA','BA','NLL','Brier','J']}))
            for c in range(8):
                take = y==c
                classes.append(dict(**meta, class_id=c, n=int(take.sum()), net=int(diff[take].sum()),
                    delta_recall=100*float(diff[take].mean())))
            for unit in ['fold','port','pass_group','valid_vh']:
                values = a.index.get_level_values('fold').to_numpy() if unit=='fold' else a[unit].to_numpy()
                for value in pd.unique(values):
                    take = values==value
                    nm, rm = u.parent.assessment(y[take], score[take]), u.parent.assessment(y[take], before[take])
                    strata.append(dict(**meta, unit=unit, name=value, n=int(take.sum()), net=int(diff[take].sum()),
                        **{f'delta_{k}': nm[k]-rm[k] for k in ['OA','BA','NLL','Brier']}))
    for name, rows in [('paired_metrics',pairs),('paired_classes',classes),('stratified',strata)]:
        pd.DataFrame(rows).to_csv(HERE/f'{stage}_{name}.csv', index=False)
    role = 'validation' if stage=='source' else 'test'
    selected = pd.DataFrame(pairs)
    selected = selected[selected.role.eq(role)]
    print(selected.to_string(index=False), flush=True)
    if stage=='source':
        lines = ['# 源端单侧必要性结果', '',
            '18新头全部按原源J选点；two_sided9直接复用。915折暴露/816唯一对象，不是独立确认。目标尚未推理。', '',
            '|seed|臂减参照|净数|ΔOA pp|ΔBA pp|ΔNLL|ΔBrier|', '|---|---|---:|---:|---:|---:|---:|']
        for q in selected.to_dict('records'):
            lines.append(f"|{q['seed']}|{q['arm']}-{q['reference']}|{q['net']:+d}|{q['delta_OA']:+.4f}|{q['delta_BA']:+.4f}|{q['delta_NLL']:+.6f}|{q['delta_Brier']:+.6f}|")
        lines.extend(['', 'source-only scaler，缺失侧及中心槽严格零；同266名义容量但有效输入维数减少。',
            '保留全部选点及正负结果，不从这些配对另选臂/seed，不改τ；报告冻结后才固定历史test。'])
        (HERE/'SOURCE_FINDINGS_ZH.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    else:
        cases = render(p)
        print(pd.DataFrame(cases)[['instance_id','reason','base_prediction','two_sided_prediction','high_prediction','low_prediction']].to_string(index=False), flush=True)
    event('analysis_complete', stage=stage)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage', choices=['source','test'])
    analyze(parser.parse_args().stage)
