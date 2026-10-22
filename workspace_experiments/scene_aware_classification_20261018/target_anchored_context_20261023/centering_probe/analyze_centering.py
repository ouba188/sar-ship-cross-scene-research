"""Paired parent comparison and descriptive, fixed-checkpoint decomposition."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from run_uncentered import HERE, PARENT, EXP, ARMS, Anchored, metrics, SEEDS
sys.path.insert(0, str(EXP))
from analyze_expanded import clustered_ci


def margin(score, y):
    other = np.asarray(score).copy()
    other[np.arange(len(y)), y] = -np.inf
    return score[np.arange(len(y)), y]-other.max(1)


def native_panel(row, targets, edges):
    from osgeo import gdal
    gdal.UseExceptions()
    gdal.SetConfigOption('GDAL_PAM_ENABLED', 'NO')
    t = targets.set_index('instance_id').loc[row.instance_id]
    canvas = Image.new('RGB', (1250, 690), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 17)
    draw.text((10, 8), f'{row.instance_id} {row.arm} {row.kind} seed29 true={row.class_id}', fill='black', font=font)
    draw.text((10, 32), f'base={row.base_prediction} centered={row.parent_prediction} uncentered={row.prediction}', fill='black', font=font)
    for k, pol in enumerate(('vv', 'vh')):
        with Image.open(EXP/'readout/native_targets'/f'{row.instance_id}_{pol}.png') as im:
            native = im.convert('RGB')
        canvas.paste(native, (10+k*200, 105))
        draw.text((10+k*200, 75), f'Target {pol.upper()} native {native.width}x{native.height}', fill='black', font=font)
    for j, e in enumerate(edges[edges.instance_id.eq(row.instance_id)].sort_values('slot').head(3).itertuples()):
        x = 420+270*j
        draw.text((x, 75), f'{e.detection_id} {e.distance_m:.0f}m', fill='black', font=font)
        for k, pol in enumerate(('vv', 'vh')):
            ds = gdal.Open(getattr(e, 'raster_'+pol))
            win = json.loads(getattr(e, 'read_window_'+pol))
            im = Image.fromarray(ds.ReadAsArray(*win)).convert('RGB')
            assert im.width <= 260 and im.height <= 240
            draw.text((x, 100+250*k), pol.upper(), fill='black', font=font)
            canvas.paste(im, (x, 128+250*k))
            ds = None
    draw.text((10, 640), 'Unchanged native pixels; first three neighbors displayed, model uses all valid nodes.', fill='black', font=font)
    draw.text((10, 663), '0=bulk, 1=fishing, 2=general. No physical mechanism inferred from brightness.', fill='black', font=font)
    path = HERE/f'case_{row.instance_id}.png'
    canvas.save(path)
    return path


def main():
    torch.set_num_threads(2)
    t = pd.read_csv(EXP/'readout/targets.csv', dtype={'mmsi': str})
    p = pd.read_csv(HERE/'predictions.csv')
    parent = pd.concat([pd.read_csv(PARENT/'predictions.csv'), pd.read_csv(PARENT/'control_predictions.csv')])
    all_pairs, paired_metrics, classes, acquisitions = [], [], [], []
    for arm, reference in ARMS.items():
        aa = p[p.arm.eq(arm)].copy()
        bb = parent[parent.arm.eq(reference)].rename(columns={
            'prediction':'parent_prediction', 'base_prediction':'parent_base_prediction',
            **{f's{c}':f'parent_s{c}' for c in range(3)}})
        aa = aa.merge(bb[['seed','fold','instance_id','class_id','parent_prediction','parent_base_prediction',
                         'parent_s0','parent_s1','parent_s2']],
                      on=['seed','fold','instance_id','class_id'], validate='one_to_one')
        assert len(aa) == len(t)*3 and aa.base_prediction.eq(aa.parent_base_prediction).all()
        aa = aa.merge(t[['instance_id','mmsi','pass_group','port']], on='instance_id', validate='many_to_one')
        aa['correct'] = aa.prediction.eq(aa.class_id)
        aa['base_correct'] = aa.base_prediction.eq(aa.class_id)
        aa['parent_correct'] = aa.parent_prediction.eq(aa.class_id)
        aa['net_vs_base'] = aa.correct.astype(int)-aa.base_correct.astype(int)
        aa['net_vs_parent'] = aa.correct.astype(int)-aa.parent_correct.astype(int)
        aa['kind'] = np.select([aa.net_vs_base.eq(1), aa.net_vs_base.eq(-1), aa.correct],
                               ['rescued', 'harmed', 'CC'], default='WW')
        aa['margin'] = margin(aa[['s0','s1','s2']].to_numpy(), aa.class_id.to_numpy())
        aa['parent_margin'] = margin(aa[['parent_s0','parent_s1','parent_s2']].to_numpy(), aa.class_id.to_numpy())
        aa['base_margin'] = margin(aa[['b0','b1','b2']].to_numpy(), aa.class_id.to_numpy())
        all_pairs.append(aa)
        for seed, part in aa.groupby('seed'):
            for scope, q in [('all',part), ('external',part[part.has_external]), ('no_external',part[~part.has_external])]:
                for ref in ('base', 'parent'):
                    b = q[ref+'_correct']
                    delta = q.correct.astype(float)-b
                    ba = delta.groupby(q.class_id).mean().mean()*100
                    paired_metrics.append(dict(arm=arm, reference=reference if ref == 'parent' else ref,
                        seed=seed, scope=scope, n=len(q), delta_OA=float(delta.mean()*100), delta_BA=float(ba),
                        rescued=int((q.correct & ~b).sum()), harmed=int((~q.correct & b).sum()),
                        net=int(delta.sum()), agreement=float(q.prediction.eq(q.parent_prediction if ref == 'parent' else q.base_prediction).mean()*100)))
        for (seed, c), q in aa.groupby(['seed','class_id']):
            classes.append(dict(arm=arm, seed=seed, class_id=c, n=len(q),
                recall=float(q.correct.mean()*100), parent_recall=float(q.parent_correct.mean()*100),
                net_vs_base=int(q.net_vs_base.sum()), net_vs_parent=int(q.net_vs_parent.sum())))
        for (seed, acq), q in aa.groupby(['seed','pass_group']):
            acquisitions.append(dict(arm=arm, seed=seed, pass_group=acq, n=len(q),
                net_vs_base=int(q.net_vs_base.sum()), net_vs_parent=int(q.net_vs_parent.sum())))
    pairs = pd.concat(all_pairs, ignore_index=True)
    assert np.isfinite(pairs[['s0','s1','s2','margin','parent_margin','base_margin']]).all().all()
    pairs.to_csv(HERE/'paired_predictions.csv', index=False)
    pd.DataFrame(paired_metrics).to_csv(HERE/'paired_metrics.csv', index=False)
    pd.DataFrame(classes).to_csv(HERE/'class_effects.csv', index=False)
    pd.DataFrame(acquisitions).to_csv(HERE/'acquisition_effects.csv', index=False)
    intervals = []
    for arm, q in pairs.groupby('arm'):
        for ref in ('base','parent'):
            avg = q.groupby('instance_id')['net_vs_'+ref].mean().reindex(t.instance_id).to_numpy()
            for block in ('pass_group','mmsi'):
                intervals.append(dict(arm=arm, reference=ARMS[arm] if ref == 'parent' else ref,
                    block=block, delta_OA=float(avg.mean()*100),
                    delta_BA=float(pd.Series(avg).groupby(t.class_id).mean().mean()*100),
                    **clustered_ci(t, avg, block)))
    pd.DataFrame(intervals).to_csv(HERE/'paired_intervals.csv', index=False)

    existing = {}
    for pattern in ('case_pair_*.png', 'control_case_pair_*.png'):
        for path in sorted(PARENT.glob(pattern)):
            existing['pair_'+path.stem.split('pair_', 1)[1]] = path
    chosen = []
    edges = pd.read_csv(EXP/'readout/neighbors.csv')
    for arm in ARMS:
        q = pairs[pairs.arm.eq(arm) & pairs.seed.eq(29) & pairs.active].copy()
        for kind in ('rescued','harmed','CC','WW'):
            candidates = q[q.kind.eq(kind)].copy()
            assert len(candidates), (arm, kind)
            candidates['existing_image'] = candidates.instance_id.isin(existing)
            row = candidates.sort_values(['existing_image','instance_id'], ascending=[False,True]).iloc[0]
            path = existing.get(row.instance_id)
            reused = path is not None
            if path is None:
                path = native_panel(row, t, edges)
            record = row.drop(labels=['existing_image']).to_dict()
            record.update(panel=str(path), reused_parent_panel=reused,
                selection='seed29, active, outcome_vs_base; prefer existing parent panel then ascending ID',
                native_vv=str(EXP/'readout/native_targets'/f'{row.instance_id}_vv.png'),
                native_vh=str(EXP/'readout/native_targets'/f'{row.instance_id}_vh.png'))
            chosen.append(record)
    cases = pd.DataFrame(chosen)
    cases.to_csv(HERE/'review_cases.csv', index=False)

    summaries, group_rows, replay, rival_rows, scale_rows = [], [], [], [], []
    loc = dict(zip(t.instance_id, range(len(t))))
    for fold in range(6):
        d = dict(np.load(PARENT/f'fold{fold}_inputs.npz'))
        cache = np.load(HERE/f'fold{fold}_inputs.npz')
        np.testing.assert_array_equal(cache['base'], d['base'])
        np.testing.assert_array_equal(cache['oof'], d['oof'])
        test = d['test']
        qc = d['q'][test]
        origin = cache['origin']
        for g in range(32):
            scale_rows.append(dict(fold=fold, group=g,
                origin_score_rms=float(np.sqrt(np.mean(origin[g]**2))),
                source_centered_rms=float(np.sqrt(np.mean(d['q'][d['train'],g]**2))),
                source_raw_rms=float(np.sqrt(np.mean(cache['qraw'][d['train'],g]**2)))))
        for arm, ref in ARMS.items():
            env0 = d['env'] if arm == 'uncentered_external' else np.load(PARENT/f'fold{fold}_own_input.npz')['env']
            active = d['has'][test] if arm == 'uncentered_external' else np.ones(len(test), bool)
            env = torch.tensor(env0[test], dtype=torch.float32, device='cuda')
            for seed in SEEDS:
                models = []
                for root, name in [(HERE,arm),(PARENT,ref)]:
                    model = Anchored().cuda().eval()
                    model.load_state_dict(torch.load(root/'models'/f'fold{fold}_{name}_{seed}.pt', weights_only=True, map_location='cpu'))
                    with torch.no_grad():
                        models.append((.5*torch.tanh(model.net(env))).cpu().numpy())
                a, ac = models
                gate_term = (a-ac)[:,:,None]*qc*active[:,None,None]
                origin_term = a[:,:,None]*origin[None]*active[:,None,None]
                q = pairs[pairs.fold.eq(fold)&pairs.arm.eq(arm)&pairs.seed.eq(seed)].set_index('instance_id').loc[t.iloc[test].instance_id]
                diff = q[['s0','s1','s2']].to_numpy()-q[['parent_s0','parent_s1','parent_s2']].to_numpy()
                error = float(np.abs(gate_term.sum(1)+origin_term.sum(1)-diff).max())
                assert error < 1e-5, error
                replay.append(dict(fold=fold, arm=arm, seed=seed, decomposition_error=error, baseline_bitwise_equal=True))
                for g in range(32):
                    summaries.append(dict(fold=fold, arm=arm, seed=seed, group=g, n=int(active.sum()),
                        mean_abs_gate=float(np.abs(a[active,g]).mean()),
                        parent_mean_abs_gate=float(np.abs(ac[active,g]).mean()),
                        origin_score_rms=float(np.sqrt(np.mean(origin_term[active,g]**2))),
                        changed_gate_score_rms=float(np.sqrt(np.mean(gate_term[active,g]**2)))))
                for case in cases[cases.fold.eq(fold)&cases.arm.eq(arm)&cases.seed.eq(seed)].itertuples():
                    i = loc[case.instance_id]
                    j = int(np.flatnonzero(test == i)[0])
                    c = int(t.iloc[i].class_id)
                    rival = int(np.argmax(np.where(np.arange(3) == c, -np.inf, d['base'][i])))
                    for rival_class in range(3):
                        if rival_class != c:
                            rival_rows.append(dict(instance_id=case.instance_id, arm=arm, seed=seed, fold=fold,
                                true_class=c, rival=rival_class,
                                changed_gate_margin=float((gate_term[j,:,c]-gate_term[j,:,rival_class]).sum()),
                                origin_margin=float((origin_term[j,:,c]-origin_term[j,:,rival_class]).sum()),
                                parent_margin=float(q.iloc[j]['parent_s'+str(c)]-q.iloc[j]['parent_s'+str(rival_class)]),
                                raw_margin=float(q.iloc[j]['s'+str(c)]-q.iloc[j]['s'+str(rival_class)])))
                    for g in range(32):
                        group_rows.append(dict(instance_id=case.instance_id, arm=arm, seed=seed, fold=fold,
                            group=g, true_class=c, fixed_base_rival=rival,
                            gate=float(a[j,g]), parent_gate=float(ac[j,g]),
                            centered_evidence=float(qc[j,g,c]-qc[j,g,rival]),
                            origin_evidence=float(origin[g,c]-origin[g,rival]),
                            changed_gate_margin=float(gate_term[j,g,c]-gate_term[j,g,rival]),
                            origin_margin=float(origin_term[j,g,c]-origin_term[j,g,rival])))
    pd.DataFrame(summaries).to_csv(HERE/'modulation_summary.csv', index=False)
    pd.DataFrame(group_rows).to_csv(HERE/'case_group_contributions.csv', index=False)
    pd.DataFrame(rival_rows).to_csv(HERE/'case_rival_decomposition.csv', index=False)
    pd.DataFrame(scale_rows).to_csv(HERE/'origin_scale.csv', index=False)
    raw_stem = np.load(EXP/'readout/features.npz')['target']
    channel_rows = []
    for fold in range(6):
        source = np.load(HERE/'origins'/f'fold{fold}_outer.npz')
        if 'stem_W' not in source:
            continue
        std = raw_stem[source['fit']].std(0, dtype=np.float64)
        for c in range(3):
            for rival in range(c+1, 3):
                w = source['stem_W'][c]-source['stem_W'][rival]
                for col in range(raw_stem.shape[1]):
                    channel_rows.append(dict(fold=fold, class_a=c, class_b=rival, channel=col,
                        source_mean=float(source['stem_h0'][col]), source_std=float(std[col]),
                        W_difference=float(w[col]), origin_difference=float(w[col]*source['stem_h0'][col])))
    pd.DataFrame(channel_rows).to_csv(HERE/'stem_origin_channels.csv', index=False)
    pd.DataFrame(replay).to_csv(HERE/'decomposition_verification.csv', index=False)
    summary = dict(paired_rows=len(pairs), paired_heads=36, case_rows=len(cases),
        unique_images=cases.panel.nunique(), reused_parent_case_rows=int(cases.reused_parent_panel.sum()),
        max_decomposition_error=max(v['decomposition_error'] for v in replay),
        note='CI uses seed-averaged paired differences, separate pass and MMSI block bootstrap; not joint.',
        image_note='Reused parent panel captions refer to parent models. Current case predictions/margins are in review_cases.csv.')
    (HERE/'analysis_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(pd.DataFrame(paired_metrics).loc[lambda q:q.scope.eq('all')].to_string(index=False), flush=True)
    print(pd.DataFrame(intervals).to_string(index=False), flush=True)
    print(cases[['arm','kind','instance_id','base_prediction','parent_prediction','prediction','panel']].to_string(index=False), flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
