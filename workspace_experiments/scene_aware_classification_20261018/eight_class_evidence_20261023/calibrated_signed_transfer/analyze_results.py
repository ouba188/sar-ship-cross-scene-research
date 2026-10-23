"""Descriptive fixed-prediction strata, paired intervals, and native review panels."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
import argparse
import json
import sys
import time
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_training import HERE, PARENT, assessment, load_table
sys.path.insert(0, str(PARENT))
from prepare_features import OLD, crop_pad, masks_for


def metrics(g):
    y = g.class_id.to_numpy()
    before = assessment(y, g[[f'b{c}' for c in range(8)]].to_numpy())
    after = assessment(y, g[[f's{c}' for c in range(8)]].to_numpy())
    return dict(**after, **{f'base_{k}': v for k, v in before.items() if k != 'n'},
                net=int(g.net.sum()), rescued=int(g.net.eq(1).sum()), harmed=int(g.net.eq(-1).sum()),
                delta_OA=after['OA']-before['OA'], delta_BA=after['BA']-before['BA'],
                delta_NLL=after['NLL']-before['NLL'], delta_Brier=after['Brier']-before['Brier'],
                mmsi=g.mmsi.nunique(), passes=g.pass_group.nunique(), present_classes=g.class_id.nunique())


def intervals(g, diff, label):
    rows = []
    y = g.class_id.to_numpy()
    for block in ['pass_group', 'mmsi']:
        codes, names = pd.factorize(g[block], sort=True)
        counts = np.zeros((len(names), 8))
        nets = np.zeros_like(counts)
        np.add.at(counts, (codes, y), 1)
        np.add.at(nets, (codes, y), diff)
        rng = np.random.default_rng(20261023)
        oa, ba = [], []
        for _ in range(2000):
            ix = rng.integers(0, len(names), len(names))
            den, num = counts[ix].sum(0), nets[ix].sum(0)
            oa.append(100*num.sum()/den.sum())
            if (den > 0).all():
                ba.append(100*np.mean(num/den))
        rows.append(dict(comparison=label, block=block, blocks=len(names), draws=2000, bootstrap_seed=20261023,
                         missing_class_draws=2000-len(ba), net=int(diff.sum()), delta_OA=100*diff.mean(),
                         delta_BA=100*np.mean([diff[y == c].mean() for c in range(8)]),
                         OA_low=float(np.quantile(oa, .025)), OA_high=float(np.quantile(oa, .975)),
                         BA_low=float(np.quantile(ba, .025)), BA_high=float(np.quantile(ba, .975))))
    return rows


def render(stage, p, show_used_polarization_moments=False):
    out = HERE/stage
    (out/'panels').mkdir(exist_ok=True)
    frame = load_table().set_index('instance_id')
    snapshots = {s['obs_id']: s for s in json.loads((OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    coverage = pd.read_csv(PARENT/'coverage.csv').set_index(['instance_id', 'pol'])
    cache = dict(np.load(HERE/'signed_features.npz'))
    lookup = {s: i for i, s in enumerate(cache['instance_ids'])}
    rows, missing = [], []
    for arm, g in p[p.seed.eq(17)].groupby('arm'):
        selections = []
        for outcome in ['rescued', 'harmed', 'CC', 'WW']:
            hits = g[g.outcome.eq(outcome)].sort_values('instance_id')
            if len(hits):
                selections.append(hits.iloc[0])
            else:
                missing.append(dict(arm=arm, outcome=outcome))
        if len(selections) < 4:
            for outcome in ['CC', 'WW']:
                selections.extend(row for _, row in g[g.outcome.eq(outcome)].sort_values('instance_id').iloc[1:2].iterrows())
        for pred in selections:
            ident = pred.instance_id
            row = frame.loc[ident]
            truth = int(row.class_id)
            base = pred[[f'b{c}' for c in range(8)]].to_numpy(float)
            score = pred[[f's{c}' for c in range(8)]].to_numpy(float)
            wrong = max((c for c in range(8) if c != truth), key=lambda c: base[c])
            ri = list(cache['reader_names']).index(f'fold{int(pred.fold)}_outer')
            mom = cache['moments_pol'][ri, lookup[ident]]
            moment_note = 'Observed per-polarization moments.'
            if show_used_polarization_moments and arm == 'aggregate_repeated':
                valid = cache['valid'][lookup[ident]].astype(bool)
                mom = mom.sum(0)[None] / valid.sum() * valid[:, None, None]
                moment_note = 'Displayed moments are aggregate-repeated INPUT slots, not original VV/VH moments.'
            fig, axes = plt.subplots(2, 3, figsize=(15, 11), layout='constrained')
            fig.get_layout_engine().set(h_pad=0.12, hspace=0.08)
            for j, pol in enumerate(['vv', 'vh']):
                view = snapshots[int(row.obs_id)]['images'][pol]
                with Image.open(row[f'image_{pol}']) as im:
                    raw = np.array(im.convert('L'))
                pixels, masks, place = crop_pad(raw, masks_for(view), 160)
                support = masks.sum(0)
                weight = F.interpolate(torch.tensor(support)[None, None], size=(10, 10), mode='area')[0, 0].numpy()
                used = bool(coverage.loc[(ident, pol), 'valid'])
                axes[j, 0].imshow(raw, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
                axes[j, 0].set_title(f'{pol.upper()} original {raw.shape[1]}x{raw.shape[0]} px')
                axes[j, 1].imshow(pixels, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
                axes[j, 1].set_title('Actual 160 crop/pad'+('' if used else ' [NOT USED]'))
                axes[j, 2].pcolormesh(np.arange(11)*16, np.arange(11)*16, weight, cmap='Greys_r', vmin=0, vmax=1,
                                     edgecolors='#666666', linewidth=.4, shading='flat')
                axes[j, 2].set(xlim=(0, 160), ylim=(160, 0), aspect='equal')
                axes[j, 2].set_title(f'10x10 footprint; valid={used}; coverage={support.mean():.3f}\n'
                    f'sigma true/wrong={mom[j, truth, 0]:.3f}/{mom[j, wrong, 0]:.3f}; '
                    f'kappa={mom[j, truth, 1]:+.3f}/{mom[j, wrong, 1]:+.3f}', fontsize=10)
                prior = view['priors']
                target = prior.get('target_index')
                detections = prior.get('detections', [])
                if target is not None and 0 <= target < len(detections):
                    corners = np.array(detections[target]['corners'])
                    points = np.r_[corners, corners[:1]]
                    axes[j, 0].plot(points[:, 0], points[:, 1], color='#13c74e', lw=1)
                    points = points-[place['crop_x'], place['crop_y']]+[place['pad_left'], place['pad_top']]
                    for ax in axes[j, 1:]:
                        ax.plot(points[:, 0], points[:, 1], color='#13c74e', lw=1)
                for ax in axes[j]:
                    ax.set_xlabel('pixel x')
                    ax.set_ylabel('pixel y')
            fig.suptitle(f'{stage} / {arm} / seed17 / {pred.outcome} / {ident}\n'
                f'{row.class_name} / {row.port} / fold{int(pred.fold)} / epoch{int(pred.epoch)}; '
                f'prediction {int(pred.base_prediction)} -> {int(pred.prediction)}\n'
                f'true {truth} vs fixed wrong {wrong}: margin {base[truth]-base[wrong]:+.5f} -> '
                f'{score[truth]-score[wrong]:+.5f}; correction {(score-base)[truth]-(score-base)[wrong]:+.5f}\n'
                'Green: recorded OBB. Area footprint affects pooling, not image pixels; convolution RF may mix padding.\n'
                + moment_note, fontsize=11)
            path = out/'panels'/f'{arm}_{pred.outcome}_{ident}.png'
            fig.savefig(path, dpi=110)
            plt.close(fig)
            r = dict(stage=stage, arm=arm, seed=17, instance_id=ident, outcome=pred.outcome, fold=int(pred.fold),
                     class_id=truth, class_name=row.class_name, port=row.port, base_prediction=int(pred.base_prediction),
                     prediction=int(pred.prediction), fixed_wrong=wrong, base_margin=float(base[truth]-base[wrong]),
                     margin=float(score[truth]-score[wrong]), image_vv=row.image_vv, image_vh=row.image_vh,
                     vv_valid=bool(pred.valid_vv), vh_valid=bool(pred.valid_vh), panel=str(path),
                     selection='seed17 first ID per outcome; second CC/WW only if absent flip')
            for j, pol in enumerate(['vv', 'vh']):
                for c, label in [(truth, 'true'), (wrong, 'wrong')]:
                    r[f'{pol}_sigma_{label}'], r[f'{pol}_kappa_{label}'] = mom[j, c]
            rows.append(r)
    pd.DataFrame(rows).to_csv(out/'review_cases.csv', index=False)
    return [r['panel'] for r in rows], missing


def main(stage, show_used_polarization_moments=False):
    tick = time.perf_counter()
    torch.set_num_threads(2)
    out = HERE/stage
    p = pd.read_csv(out/'predictions.csv', dtype={'mmsi': str})
    assert np.isfinite(p[[f's{c}' for c in range(8)]].to_numpy()).all()
    assert np.array_equal(p.prediction, p[[f's{c}' for c in range(8)]].to_numpy().argmax(1))
    strata = []
    for (role, arm, seed), g in p.groupby(['role', 'arm', 'seed']):
        for col in ['fold', 'class_id', 'valid_vh', 'port', 'pass_group', 'product']:
            for val, h in g.groupby(col):
                strata.append(dict(stage=stage, role=role, arm=arm, seed=seed, unit=col, name=val, **metrics(h)))
        for (c, vh), h in g.groupby(['class_id', 'valid_vh']):
            strata.append(dict(stage=stage, role=role, arm=arm, seed=seed, unit='class_vh', name=f'{c}/{vh}', **metrics(h)))
    s = pd.DataFrame(strata)
    s.to_csv(out/'stratified_metrics.csv', index=False)
    for unit, filename in [('fold', 'fold_metrics'), ('class_id', 'class_metrics'), ('valid_vh', 'vh_metrics')]:
        s[s.unit.eq(unit)].to_csv(out/f'{filename}.csv', index=False)
    test = p[p.role.eq('test')]
    ci = []
    for (arm, seed), g in test.groupby(['arm', 'seed']):
        for row in intervals(g, g.net.to_numpy(), f'{arm}-'+('A' if stage == 'calibration' else 'B')):
            ci.append(dict(stage=stage, seed=seed, **row))
    if stage == 'increment':
        for seed in [17, 29, 43]:
            signed = test[test.seed.eq(seed)&test.arm.eq('signed_only')].set_index('instance_id').sort_index()
            for other in ['score_only', 'conditional_std']:
                ref = test[test.seed.eq(seed)&test.arm.eq(other)].set_index('instance_id').loc[signed.index]
                diff = signed.net.to_numpy()-ref.net.to_numpy()
                for row in intervals(signed, diff, 'signed_only-'+other):
                    ci.append(dict(stage=stage, seed=seed, **row))
        calibration = pd.read_csv(HERE/'calibration/predictions.csv')
        key = ['role', 'fold', 'seed', 'instance_id']
        expected = calibration.set_index(key)
        errors = []
        for arm, g in p.groupby('arm'):
            g = g.set_index(key)
            error = float(abs(g[[f'b{c}' for c in range(8)]].to_numpy()-
                              expected.loc[g.index, [f's{c}' for c in range(8)]].to_numpy()).max())
            # Float32 GEMM can differ between full-cohort and subset batch sizes.
            changes = int(np.sum(g.base_prediction.to_numpy() != expected.loc[g.index, 'prediction'].to_numpy()))
            assert error < 2e-6 and changes == 0
            errors.append(dict(arm=arm, frozen_C_replay_error=error, top1_changes=changes,
                               reason='full-cohort versus subset float32 inference batches'))
        pd.DataFrame(errors).to_csv(out/'frozen_C_replays.csv', index=False)
    pd.DataFrame(ci).to_csv(out/'paired_intervals.csv', index=False)
    test.groupby(['arm', 'seed', 'class_id', 'base_prediction', 'prediction']).size().rename('n').reset_index().to_csv(out/'transitions.csv', index=False)
    panels, missing = render(stage, test, show_used_polarization_moments)
    summary = dict(stage=stage, panels=panels, missing_outcomes=missing, seconds=time.perf_counter()-tick,
                   bootstrap_scope='paired fixed predictions; no refitting or selection uncertainty',
                   stratum_BA='macro recall over classes present within stratum',
                   imagery_review_pending=True)
    (out/'analysis_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['calibration', 'increment'])
    main(parser.parse_args().stage)
