"""One deterministic within-score-block intervention; no model fitting."""
import os
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(name, '2')
import sys
sys.dont_write_bytecode = True
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
INTEGRATION = HERE.parent / 'full_model_integration'
sys.path.insert(0, str(INTEGRATION))
import run_integration as source
parent, ref = source.parent, source.ref


def match_pairs(base, valid, ids, mmsi, mean, scale):
    """Greedy shortest disjoint edges, without access to ground truth."""
    n = len(base)
    z = (base - mean) / scale
    pred = base.argmax(1)
    donor = np.arange(n)
    distance = np.full(n, np.nan)
    sizes = np.zeros(n, int)
    candidates = np.zeros(n, int)
    reason = np.full(n, '', dtype=object)
    blocks = pd.DataFrame(dict(pred=pred, vv=valid[:, 0], vh=valid[:, 1]))
    for _, indices in blocks.groupby(['pred', 'vv', 'vh'], sort=True).groups.items():
        ix = np.array(sorted(indices, key=lambda i: str(ids[i])))
        sizes[ix] = len(ix)
        if len(ix) == 1:
            reason[ix] = 'singleton_condition_group'
            continue
        left, right = np.triu_indices(len(ix), k=1)
        eligible = mmsi[ix[left]] != mmsi[ix[right]]
        left, right = left[eligible], right[eligible]
        candidates[ix] = np.bincount(np.r_[left, right], minlength=len(ix))
        dist = np.linalg.norm(z[ix[left]] - z[ix[right]], axis=1)
        order = np.lexsort((right, left, dist))
        used = np.zeros(len(ix), bool)
        for k in order:
            a, b = left[k], right[k]
            if used[a] or used[b]:
                continue
            used[a] = used[b] = True
            i, j = ix[a], ix[b]
            donor[i], donor[j] = j, i
            distance[i] = distance[j] = dist[k]
        for i in ix[~used]:
            reason[i] = ('no_different_mmsi_candidate' if candidates[i] == 0
                         else 'eligible_candidates_already_paired')
    matched = donor != np.arange(n)
    assert np.array_equal(donor[donor], np.arange(n))
    assert np.array_equal(np.sort(donor), np.arange(n))
    assert np.array_equal(pred[matched], pred[donor[matched]])
    assert np.array_equal(valid[matched], valid[donor[matched]])
    assert np.all(mmsi[matched] != mmsi[donor[matched]])
    assert np.isfinite(distance[matched]).all() and np.all(reason[~matched] != '')
    return donor, distance, sizes, candidates, reason


def evaluate(model, ck, base, moments, valid):
    x, carrier, _, _ = source.inputs('free_real', base, moments, valid, ck['mu'], ck['scale'])
    with torch.inference_mode():
        delta = model(x, carrier).numpy().astype(float)
    return base + delta


def main(centered=False):
    started = time.perf_counter()
    assert (HERE / 'PROTOCOL_ZH.md').exists()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    table = parent.load_table()
    cache = dict(np.load(parent.HERE / 'signed_features.npz'))
    valid = cache['valid'].astype(bool)
    saved = pd.read_csv(INTEGRATION / 'predictions.csv', dtype={'mmsi': str})
    saved = saved[saved.arm.eq('free_real')].set_index(['fold', 'seed', 'role', 'instance_id'])
    frames, checks, scales = [], [], []
    for fold in range(3):
        d, _, _ = parent.load_fold(fold, table, cache)
        outer, _ = ref.load_pol(fold, table, cache, d)
        for oldseed, seed in source.SEED_PAIRS:
            base, train_base = source.load_old(fold, oldseed, table, d)
            source_match = train_base[d['train']]
            if centered:
                source_match = source_match - source_match.mean(1, keepdims=True)
            mean = source_match.mean(0)
            raw_scale = source_match.std(0)
            scale = raw_scale.clip(1e-8)
            for c in range(8):
                scales.append(dict(fold=fold, seed=seed, oldseed=oldseed, class_id=c,
                                   train_rows=len(d['train']), mean=mean[c], raw_std=raw_scale[c], scale=scale[c]))
            path = INTEGRATION / 'models' / f'fold{fold}_free_real_seed{seed}.pt'
            ck, model = source.restore(path)
            for role in ['validation', 'test']:
                ix = d[role]
                g = table.iloc[ix][['instance_id', 'obs_id', 'class_id', 'class_name', 'mmsi',
                                     'product', 'port', 'pass_group']].copy().reset_index(drop=True)
                b, moments, v = base[ix], outer[ix], valid[ix]
                original = evaluate(model, ck, b, moments, v)
                for k, value in dict(fold=fold, oldseed=oldseed, seed=seed, role=role, epoch=ck['epoch']).items():
                    g[k] = value
                expected = saved.loc[pd.MultiIndex.from_frame(g[['fold', 'seed', 'role', 'instance_id']])]
                saved_score = expected[[f's{c}' for c in range(8)]].to_numpy()
                saved_base = expected[[f'b{c}' for c in range(8)]].to_numpy()
                original_error = float(abs(original - saved_score).max())
                base_error = float(abs(b - saved_base).max())
                np.testing.assert_allclose(original, saved_score, atol=1e-7, rtol=0)
                np.testing.assert_allclose(b, saved_base, atol=1e-7, rtol=0)
                assert np.array_equal(original.argmax(1), expected.prediction)
                assert np.array_equal(b.argmax(1), expected.base_prediction)

                match_base = b - b.mean(1, keepdims=True) if centered else b
                assert np.array_equal(match_base.argmax(1), b.argmax(1))
                donor, distance, sizes, candidates, reason = match_pairs(
                    match_base, v, g.instance_id.to_numpy(), g.mmsi.to_numpy(), mean, scale)
                matched = donor != np.arange(len(ix))
                swapped_moments = moments[donor]
                swapped = evaluate(model, ck, b, swapped_moments, v)
                assert np.array_equal(swapped_moments[~matched], moments[~matched])
                assert np.array_equal(swapped[~matched], original[~matched])
                assert not swapped_moments[~v].any()
                assert np.isfinite(swapped).all()
                g['matched'], g['distance'], g['condition_group_size'] = matched, distance, sizes
                g['eligible_candidates'], g['unmatched_reason'] = candidates, reason
                g['valid_vv'], g['valid_vh'] = v.T
                g['donor_instance_id'] = g.instance_id.to_numpy()[donor]
                g['donor_mmsi'] = g.mmsi.to_numpy()[donor]
                g['donor_class_id'] = g.class_id.to_numpy()[donor]
                g['donor_port'] = g.port.to_numpy()[donor]
                g['donor_pass_group'] = g.pass_group.to_numpy()[donor]
                # Labels enter only here, after the label-free matching has been fixed.
                y = g.class_id.to_numpy()
                g['same_class'] = np.where(matched, y == y[donor], np.nan)
                g['same_pass'] = np.where(matched, g.pass_group.to_numpy() == g.pass_group.to_numpy()[donor], np.nan)
                g['same_port'] = np.where(matched, g.port.to_numpy() == g.port.to_numpy()[donor], np.nan)
                g['base_prediction'] = b.argmax(1)
                g['original_prediction'], g['swapped_prediction'] = original.argmax(1), swapped.argmax(1)
                for name in ['base_prediction', 'original_prediction', 'swapped_prediction']:
                    g['donor_' + name] = g[name].to_numpy()[donor]
                bc, oc, wc = b.argmax(1) == y, original.argmax(1) == y, swapped.argmax(1) == y
                g['original_net_B'] = oc.astype(int) - bc.astype(int)
                g['swapped_net_B'] = wc.astype(int) - bc.astype(int)
                g['original_minus_swapped'] = oc.astype(int) - wc.astype(int)
                g['swap_outcome'] = np.select([~oc & wc, oc & ~wc, oc & wc], ['rescued', 'harmed', 'CC'], default='WW')
                g['pair_key'] = ['|'.join(sorted([str(g.instance_id.iloc[i]), str(g.instance_id.iloc[j])]))
                                 if matched[i] else '' for i, j in enumerate(donor)]
                for c in range(8):
                    g[f'b{c}'], g[f'o{c}'], g[f'w{c}'] = b[:, c], original[:, c], swapped[:, c]
                checks.append(dict(fold=fold, seed=seed, role=role, epoch=ck['epoch'], n=len(g),
                                   matched=int(matched.sum()), pairs=int(matched.sum() // 2),
                                   original_score_max_error=original_error, base_max_error=base_error,
                                   top1_mismatches=0, permutation_and_constraints_pass=True,
                                   unmatched_unchanged=True, checkpoint=str(path)))
                frames.append(g)
                print(fold, seed, role, 'matched', int(matched.sum()), '/', len(g),
                      'original-swap net', int((oc.astype(int)-wc.astype(int)).sum()), flush=True)
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(HERE / 'predictions.csv', index=False)
    pd.DataFrame(scales).to_csv(HERE / 'matching_scales.csv', index=False)
    pd.DataFrame(checks).to_csv(HERE / 'replay_checks.csv', index=False)
    summary = dict(heads_replayed=9, role_replays=18, training_steps=0, new_models=0,
                   matching_rule='greedy shortest admissible edge; distance then lexicographic IDs',
                   matching_centered=centered,
                   total_seconds=time.perf_counter()-started,
                   original_score_max_error=max(r['original_score_max_error'] for r in checks),
                   base_max_error=max(r['base_max_error'] for r in checks), top1_mismatches=0)
    (HERE / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
