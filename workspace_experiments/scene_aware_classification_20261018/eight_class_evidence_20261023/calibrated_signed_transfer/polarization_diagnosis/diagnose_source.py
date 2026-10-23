"""Source-only polarization diagnostics; frozen checkpoint inference, no fitting."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import run_training as ref


def summarize(g):
    eligible = g[g.dual_valid & g.energy.gt(1e-12)]
    conflict = eligible[eligible.conflict]
    wrong = eligible[~eligible.B_correct]
    correct = eligible[eligible.B_correct]
    return dict(n=len(g), unique_ids=g.instance_id.nunique(), mmsi=g.mmsi.nunique(), passes=g.pass_group.nunique(),
                vh_valid=int(g.valid_vh.sum()), eligible=len(eligible), conflict=len(conflict),
                conflict_rate=float(eligible.conflict.mean()) if len(eligible) else None,
                cancel_mean=float(eligible.cancel.mean()) if len(eligible) else None,
                conflict_cancel_median=float(conflict.cancel.median()) if len(conflict) else None,
                energy_median=float(eligible.energy.median()) if len(eligible) else None,
                B_wrong=len(wrong), B_wrong_conflict=int(wrong.conflict.sum()),
                B_correct=len(correct), B_correct_conflict=int(correct.conflict.sum()),
                wrong_one_positive_sum_nonpositive=int((wrong[['d_vv','d_vh']].max(axis=1).gt(0)&wrong.d_sum.le(0)).sum()),
                rescued=int(g.outcome.eq('rescued').sum()), harmed=int(g.outcome.eq('harmed').sum()),
                conflict_rescued=int(conflict.outcome.eq('rescued').sum()), conflict_harmed=int(conflict.outcome.eq('harmed').sum()))


def main():
    torch.set_num_threads(2)
    table = ref.load_table()
    cache = dict(np.load(ROOT/'signed_features.npz'))
    names = list(cache['reader_names'])
    frames = []
    for fold in range(3):
        d, outer, oof = ref.load_fold(fold, table, cache)
        pol_outer = cache['moments_pol'][names.index(f'fold{fold}_outer')]
        pol_oof = np.zeros_like(pol_outer)
        for inner in range(4):
            name = f'fold{fold}_inner{inner}'
            held = np.load(ref.PARENT/'baseline/models'/f'{name}.npz')['held']
            pol_oof[held] = cache['moments_pol'][names.index(name), held]
        assert np.allclose(pol_oof[d['train']].sum(1), oof[d['train']], atol=1e-12, rtol=0)
        for seed in [17, 29, 43]:
            base, train_base, _ = ref.calibrated(fold, seed, d, outer, oof)
            ck, model = ref.restore(ref.checkpoint_path('increment', fold, seed, 'signed_only'))
            for role, b, mom, pol in [('train_oof', train_base, oof, pol_oof), ('validation', base, outer, pol_outer)]:
                ix = d['train' if role == 'train_oof' else 'validation']
                truth = table.iloc[ix].class_id.to_numpy()
                b = b[ix]
                score = ref.apply_head(ck, model, b, mom[ix])
                competing = b.copy()
                competing[np.arange(len(ix)), truth] = -np.inf
                rival = competing.argmax(1)
                kappa = pol[ix, :, :, 1]
                dv = kappa[np.arange(len(ix)), 0, truth]-kappa[np.arange(len(ix)), 0, rival]
                dh = kappa[np.arange(len(ix)), 1, truth]-kappa[np.arange(len(ix)), 1, rival]
                energy = abs(dv)+abs(dh)
                valid = cache['valid'][ix]
                eligible = valid.all(1)&(energy > 1e-12)
                cancel = np.full(len(ix), np.nan)
                cancel[eligible] = 1-abs(dv[eligible]+dh[eligible])/energy[eligible]
                g = table.iloc[ix][['instance_id','class_id','class_name','mmsi','pass_group','port','obs_id','image_vv','image_vh']].copy()
                g['row_index'], g['fold'], g['seed'], g['role'], g['epoch'] = ix, fold, seed, role, ck['epoch']
                g['valid_vv'], g['valid_vh'], g['dual_valid'] = valid[:,0], valid[:,1], valid.all(1)
                g['B_prediction'], g['S_prediction'], g['fixed_rival'] = b.argmax(1), score.argmax(1), rival
                g['actual_new_winner'] = score.argmax(1)
                g['winner_changed'] = b.argmax(1) != score.argmax(1)
                g['B_correct'], g['S_correct'] = b.argmax(1) == truth, score.argmax(1) == truth
                g['outcome'] = np.select([~g.B_correct & g.S_correct, g.B_correct & ~g.S_correct, g.B_correct & g.S_correct], ['rescued','harmed','CC'], default='WW')
                g['d_vv'], g['d_vh'], g['d_sum'], g['energy'], g['cancel'] = dv, dh, dv+dh, energy, cancel
                g['conflict'] = eligible & (dv*dh < 0)
                g['B_margin'] = b[np.arange(len(ix)), truth]-b[np.arange(len(ix)), rival]
                g['S_fixed_margin'] = score[np.arange(len(ix)), truth]-score[np.arange(len(ix)), rival]
                newwrong = score.copy()
                newwrong[np.arange(len(ix)), truth] = -np.inf
                g['S_margin'] = score[np.arange(len(ix)), truth]-newwrong.max(1)
                frames.append(g)
    events = pd.concat(frames, ignore_index=True)
    events.to_csv(HERE/'source_events.csv.gz', index=False)
    primary = events[events.seed.eq(17)].sort_values(['fold','instance_id']).drop_duplicates(['role','instance_id'])
    primary.to_csv(HERE/'source_primary.csv', index=False)
    rows = []
    for (role, seed), g in events.groupby(['role','seed']):
        unique = g.sort_values(['fold','instance_id']).drop_duplicates('instance_id')
        rows.append(dict(role=role, seed=seed, unit='unique_minfold', **summarize(unique)))
        rows.append(dict(role=role, seed=seed, unit='fold_exposures', **summarize(g)))
    summary = pd.DataFrame(rows)
    summary.to_csv(HERE/'source_summary.csv', index=False)
    strata = []
    for role, g in primary.groupby('role'):
        for col in ['class_id','valid_vh','pass_group','outcome','B_correct']:
            for value, h in g.groupby(col):
                strata.append(dict(role=role, unit=col, name=value, **summarize(h)))
    pd.DataFrame(strata).to_csv(HERE/'source_strata.csv', index=False)
    cases = []
    val = primary[primary.role.eq('validation') & primary.dual_valid & primary.energy.gt(1e-12)]
    for conflict in [True, False]:
        for correct in [True, False]:
            h = val[val.conflict.eq(conflict)&val.B_correct.eq(correct)].sort_values('instance_id')
            if len(h):
                cases.append(h.iloc[0])
    pd.DataFrame(cases).to_csv(HERE/'source_review_cases.csv', index=False)
    print(summary.to_string(index=False), flush=True)
    (HERE/'status.json').write_text(json.dumps(dict(source_complete=True, primary_seed=17,
        observation_rule='one lowest fold per role and instance_id; seeds reported separately',
        test_used=False, training=False, images_viewed=False), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
