"""Paired source diagnostics, with acquisition-cluster uncertainty summaries."""
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(HERE/'.mplcache')
from run import ROOT, SEEDS, load_view, perturbations, geo, slots, UP


def margin(score, y):
    competitor = np.where(np.arange(8)[None] == y[:, None], -np.inf, score).max(-1)
    return score[np.arange(len(y)), y]-competitor


def bootstrap(frame):
    groups = frame.groupby('pass_group').diff_vs_control.agg(['sum', 'count'])
    rng = np.random.default_rng(1024)
    take = rng.integers(0, len(groups), (2000, len(groups)))
    values = groups['sum'].to_numpy()[take].sum(1)/groups['count'].to_numpy()[take].sum(1)
    return np.quantile(values, [.025, .975])


def main():
    f = pd.read_csv(HERE/'interventions.csv', dtype={'mmsi': str})
    z = np.load(HERE/'results.npz'); scores = z['scores']; rawloc = z['rawloc']; owners = z['owners']
    y = f.class_id.to_numpy(int); raw = scores[:, rawloc]
    base = raw[:, owners]
    details = []
    for si, seed in enumerate(SEEDS):
        d = f.copy(); d['seed'] = seed
        d['before'] = base[si].argmax(-1); d['after'] = scores[si].argmax(-1)
        d['before_correct'] = d.before.eq(d.class_id); d['after_correct'] = d.after.eq(d.class_id)
        d['margin_before'] = margin(base[si], y); d['margin_after'] = margin(scores[si], y)
        d['delta_margin'] = d.margin_after-d.margin_before
        rival = np.where(np.arange(8)[None] == y[:, None], -np.inf, base[si]).argmax(-1)
        d['fixed_rival'] = rival
        d['true_score_change'] = scores[si, np.arange(len(y)), y]-base[si, np.arange(len(y)), y]
        d['rival_score_change'] = scores[si, np.arange(len(y)), rival]-base[si, np.arange(len(y)), rival]
        d['fixed_margin_change'] = d.true_score_change-d.rival_score_change
        d['rescue'] = (~d.before_correct & d.after_correct).astype(int)
        d['harm'] = (d.before_correct & ~d.after_correct).astype(int)
        details.append(d)
    detail = pd.concat(details, ignore_index=True)
    detail.to_csv(HERE/'prediction_effects.csv', index=False)
    take = detail[detail.main_sample & detail.variant.ne('raw')]
    metrics = []; classes = []
    for key, part in take.groupby(['pol', 'region', 'band', 'kind', 'seed']):
        pol, region, band, kind, seed = key
        metrics.append(dict(pol=pol, region=region, band=band, kind=kind, seed=seed, n=len(part),
            OA=100*part.after_correct.mean(), BA=100*part.groupby('class_id').after_correct.mean().mean(),
            delta_OA_pp=100*(part.after_correct.astype(float)-part.before_correct).mean(),
            delta_BA_pp=100*part.groupby('class_id').apply(lambda g:(g.after_correct.astype(float)-g.before_correct).mean(), include_groups=False).mean(),
            rescued=int(part.rescue.sum()), harmed=int(part.harm.sum()), mean_margin_change=part.delta_margin.mean()))
        for c, cell in part.groupby('class_id'):
            classes.append(dict(pol=pol, region=region, band=band, kind=kind, seed=seed, class_id=c,
                n=len(cell), rescued=int(cell.rescue.sum()), harmed=int(cell.harm.sum()), delta_margin=cell.delta_margin.mean()))
    pd.DataFrame(metrics).to_csv(HERE/'metrics.csv', index=False)
    pd.DataFrame(classes).to_csv(HERE/'classes.csv', index=False)
    mean = take.groupby(['obs_id', 'pass_group', 'pol', 'region', 'band', 'kind']).agg(delta_margin=('delta_margin', 'mean'),
        correct_after=('after_correct', 'mean'), correct_before=('before_correct', 'mean'), l2=('l2','mean'),
        clipped=('clipped_fraction','mean'), brightness=('mean_change','mean')).reset_index()
    comparisons = []; pairrows = []
    for (pol, region, band), part in mean.groupby(['pol', 'region', 'band']):
        real = part[part.kind.eq('attenuate')].set_index('obs_id')
        control = part[part.kind.ne('attenuate')].groupby('obs_id').mean(numeric_only=True)
        pairs = real[['pass_group']].copy(); pairs['diff_vs_control'] = real.delta_margin-control.delta_margin
        lo, hi = bootstrap(pairs)
        comparisons.append(dict(pol=pol, region=region, band=band, n=len(real),
            attenuation_margin=real.delta_margin.mean(), control_margin=control.delta_margin.mean(),
            difference= pairs.diff_vs_control.mean(), cluster_low=lo, cluster_high=hi,
            OA_delta_pp=100*(real.correct_after-real.correct_before).mean(),
            control_OA_delta_pp=100*(control.correct_after-control.correct_before).mean(),
            clipped=real.clipped.mean(), abs_brightness_change=real.brightness.abs().mean(),
            l2_match_error=float(abs(real.l2-control.l2).max())))
        pairrows.append(pairs.reset_index().assign(pol=pol, region=region, band=band))
    summary = pd.DataFrame(comparisons)
    summary.to_csv(HERE/'summary.csv', index=False)
    pd.concat(pairrows).to_csv(HERE/'paired_query_effects.csv', index=False)
    print(summary.to_string(index=False), flush=True)
    # Detect an accidental intervention on the untouched polarization.
    local = z['local']; other_errors = []
    for p in (0, 1):
        rows = np.flatnonzero(f.pol.eq(p))
        other_errors.append(float(abs(local[rows, :, 1-p]-local[rawloc[owners[rows]], :, 1-p]).max()))
    assert max(other_errors) < 1e-7, other_errors
    (HERE/'analysis_receipt.json').write_text(json.dumps(dict(unchanged_pol_max_errors=other_errors,
        uncertainty='exploratory paired margin difference, 2000 acquisition-cluster bootstraps, no multiplicity correction',
        configs='three existing frozen configurations, not independent new training seeds'), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
