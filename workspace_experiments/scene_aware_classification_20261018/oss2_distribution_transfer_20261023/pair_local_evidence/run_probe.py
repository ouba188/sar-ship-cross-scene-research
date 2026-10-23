"""Source-only pair-local information probe; no outer-test inference or fitting."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import itertools
import json
import time
import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import train_heads as original
from extract_features import load_batch, network, feature_map, spatial_weights
from extract_moments import weighted_moments


def pair_moments(r, weight, pairs):
    norm = weight / weight.sum(-1, keepdim=True)
    a, b = pairs.T
    diff = r[..., a, :] - r[..., b, :]
    variance = (diff.square() * norm).sum(-1)
    sigma = variance.clamp_min(0).sqrt()
    kappa = (diff * diff.abs() * norm).sum(-1) / sigma.clamp_min(1e-12)
    cross = (r[..., a, :] * r[..., b, :] * norm).sum(-1)
    va = (r[..., a, :].square() * norm).sum(-1)
    vb = (r[..., b, :].square() * norm).sum(-1)
    err = float((variance - (va + vb - 2 * cross)).abs().max())
    assert err < 1e-8
    return torch.stack([sigma, kappa], -1), cross, va + vb, err


def main():
    started = time.perf_counter()
    table = pd.read_csv(ROOT / 'cohort.csv', dtype={'mmsi': str})
    with np.load(ROOT / 'readers/fold0_inputs.npz') as z:
        d = {k: z[k] for k in z.files}
    tr, va = d['train'], d['validation']
    ix = np.sort(np.r_[tr, va])
    assert not set(ix) & set(d['test'])
    source = table.iloc[ix].reset_index(drop=True)
    lookup = {int(k): j for j, k in enumerate(ix)}
    ti, vi = np.array([lookup[int(k)] for k in tr]), np.array([lookup[int(k)] for k in va])
    with np.load(ROOT / 'calibrated/fold0_seed17.npz') as z:
        base = z['base'][ix].copy()
        base[ti] = z['train_base'][tr]
    y = source.class_id.to_numpy()
    allpairs = np.array(list(itertools.combinations(range(8), 2)))
    rows = []
    for c, e in allpairs:
        nt = [int((y[ti] == k).sum()) for k in (c, e)]
        nv = [int((y[vi] == k).sum()) for k in (c, e)]
        confusion = int(((y[ti] == c) & (base[ti].argmax(1) == e)).sum()
                        + ((y[ti] == e) & (base[ti].argmax(1) == c)).sum())
        rows.append(dict(c=int(c), d=int(e), source_confusions=confusion,
                         train_c=nt[0], train_d=nt[1], val_c=nv[0], val_d=nv[1],
                         eligible=min(nt) >= 20 and min(nv) >= 5))
    ranks = pd.DataFrame(rows).sort_values(['source_confusions', 'c', 'd'], ascending=[False, True, True])
    ranks.to_csv(HERE / 'pair_selection.csv', index=False)
    selected = ranks[ranks.eligible].head(3)
    assert len(selected) == 3
    choices = []
    for row in selected.itertuples():
        for label in [row.c, row.d]:
            eligible = source.iloc[vi][source.iloc[vi].class_id.eq(label)].copy()
            eligible['base_correct'] = base[eligible.index].argmax(1) == label
            case = eligible.sort_values(['base_correct', 'sample_id']).iloc[0]
            choices.append(dict(c=row.c, d=row.d, index=int(case.name), sample_id=case.sample_id))
    with (HERE / 'selection_frozen.json').open('w', encoding='utf-8') as f:
        json.dump(dict(fold=0, seed=17, pairs=selected[['c', 'd']].values.tolist(),
                       cases=choices, train_n=len(tr), validation_n=len(va),
                       test_rows_read=0, test_used=False), f, indent=2)
    names = ['fold0_outer'] + [f'fold0_inner{k}' for k in range(4)]
    readers = []
    assigned = np.zeros(len(source), dtype=int)
    for rid, name in enumerate(names):
        with np.load(ROOT / 'readers/models' / f'{name}.npz') as z:
            readers.append(z['W'].reshape(8, 2, 1024).transpose(1, 0, 2))
            if rid:
                assigned[[lookup[int(k)] for k in z['held']]] = rid
    assert (assigned[ti] > 0).all() and (assigned[vi] == 0).all()
    w = torch.tensor(np.stack(readers), dtype=torch.float64, device='cuda')
    shape = (len(source), 2, 28)
    pm = np.zeros(shape + (2,))
    cov = np.zeros(shape)
    uncoupled_var = np.zeros(shape)
    marginal = np.zeros((len(source), 2, 8, 2))
    means = np.zeros((len(source), 2, 8))
    model = network()
    maps, maxerr = {}, 0.
    chosen = {c['index'] for c in choices}
    last = time.perf_counter()
    with torch.inference_mode():
        for first in range(0, len(source), 16):
            stop = min(first + 16, len(source))
            x, mask = load_batch(source, first, stop)
            ids = np.repeat(assigned[first:stop], 2)
            pol = np.tile([0, 1], stop - first)
            wp = w[ids, pol]
            for rot in range(4):
                z = feature_map(model, x, rot)
                weight = spatial_weights(mask, z, rot).flatten(2).double()
                response = torch.einsum('bck,bku->bcu', wp, z.flatten(2).double())
                old, mu, r = weighted_moments(response, weight)
                pair, cc, uv, err = pair_moments(r, weight, allpairs)
                maxerr = max(maxerr, err)
                pm[first:stop] += pair.cpu().numpy().reshape(-1, 2, 28, 2) / 4
                cov[first:stop] += cc.cpu().numpy().reshape(-1, 2, 28) / 4
                uncoupled_var[first:stop] += uv.cpu().numpy().reshape(-1, 2, 28) / 4
                marginal[first:stop] += old.cpu().numpy().reshape(-1, 2, 8, 2) / 4
                means[first:stop] += mu.cpu().numpy().reshape(-1, 2, 8) / 4
                if rot == 0:
                    for local, global_ix in enumerate(range(first, stop)):
                        if global_ix in chosen:
                            maps[global_ix] = r[local*2:local*2+2].cpu().numpy().reshape(2, 8, *z.shape[-2:])
            if time.perf_counter() - last > 25:
                print('source-only maps', stop, '/', len(source), flush=True)
                last = time.perf_counter()
    with np.load(ROOT / 'moments.npz') as z:
        oldnames = list(z['reader_names'])
        old_moments, old_means = z['moments_pol'], z['mean_pol']
    reader_ix = np.array([oldnames.index(names[k]) for k in assigned])
    mm = old_moments[reader_ix, ix]
    uu = old_means[reader_ix, ix]
    replay = float(np.max(abs(mm - marginal)))
    # The source subset has a different final CUDA batch size than the full cohort.
    np.testing.assert_allclose(mm, marginal, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(uu, means, atol=1e-5, rtol=1e-5)
    np.savez_compressed(HERE / 'source_moments.npz', sample_ids=source.sample_id.to_numpy(str),
                        pairs=allpairs, pair_moments=pm, covariance=cov, uncoupled_var=uncoupled_var,
                        marginal=marginal, mean=means, base=base, train=ti, validation=vi,
                        reader=assigned)
    metrics, predictions, covariance_rows = [], [], []
    for row in selected.itertuples():
        c, e = row.c, row.d
        pairid = int(np.flatnonzero((allpairs == [c, e]).all(1))[0])
        # These directions are fixed by source labels; no target-defined competitor.
        global_x = np.column_stack([base[:, c] - base[:, e], means[:, :, c] - means[:, :, e]])
        marginal_x = np.column_stack([global_x, marginal[:, :, [c, e]].reshape(len(source), -1)])
        joint_x = np.column_stack([marginal_x, pm[:, :, pairid].reshape(len(source), -1)])
        fit = ti[np.isin(y[ti], [c, e])]
        evaluate = vi[np.isin(y[vi], [c, e])]
        for arm, features in [('global', global_x), ('marginal', marginal_x), ('same_position', joint_x)]:
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=1., class_weight='balanced',
                               solver='lbfgs', max_iter=2000, random_state=17))
            clf.fit(features[fit], (y[fit] == c).astype(int))
            prob = clf.predict_proba(features[evaluate])[:, 1]
            truth = (y[evaluate] == c).astype(int)
            prediction = (prob >= .5).astype(int)
            weights = np.array([1 / (2 * (truth == k).sum()) for k in truth])
            metrics.append(dict(c=c, d=e, arm=arm, train_n=len(fit), validation_n=len(evaluate),
                                features=features.shape[1], OA=100*accuracy_score(truth, prediction),
                                BA=100*balanced_accuracy_score(truth, prediction),
                                balanced_NLL=log_loss(truth, prob, sample_weight=weights),
                                correct=int((truth == prediction).sum())))
            for k, pr, out in zip(evaluate, prob, prediction):
                predictions.append(dict(c=c, d=e, arm=arm, sample_id=source.iloc[k].sample_id,
                    mmsi=source.iloc[k].mmsi, pass_group=source.iloc[k].pass_group,
                    true_class=int(y[k]), prediction=c if out else e, probability_c=pr))
        for role, selected_ix in [('train', fit), ('validation', evaluate)]:
            ratio = 2*cov[selected_ix, :, pairid] / np.maximum(uncoupled_var[selected_ix, :, pairid], 1e-12)
            for p, polname in enumerate(['vv', 'vh']):
                covariance_rows.append(dict(c=c, d=e, role=role, pol=polname, n=len(selected_ix),
                    median_shared_fraction=float(np.median(ratio[:, p])),
                    q10=float(np.quantile(ratio[:, p], .1)), q90=float(np.quantile(ratio[:, p], .9))))
    pd.DataFrame(metrics).to_csv(HERE / 'metrics.csv', index=False)
    pd.DataFrame(predictions).to_csv(HERE / 'predictions.csv', index=False)
    pd.DataFrame(covariance_rows).to_csv(HERE / 'covariance_effect.csv', index=False)
    (HERE / 'panels').mkdir(exist_ok=True)
    for case in choices:
        i, c, e = case['index'], case['c'], case['d']
        row = source.iloc[i]
        fig, axes = plt.subplots(2, 4, figsize=(15, 7), layout='constrained')
        for p, polname in enumerate(['vv', 'vh']):
            with Image.open(row[f'chip_path_{polname}']) as im:
                raw = np.array(im)
            axes[p, 0].imshow(raw, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
            axes[p, 0].set_title(f'{polname.upper()} native {raw.shape[1]}x{raw.shape[0]}')
            fields = [maps[i][p, c], maps[i][p, e], maps[i][p, c] - maps[i][p, e]]
            footprint = np.zeros((224, 224))
            h, width = raw.shape
            left, top = (224-width)//2, (224-h)//2
            footprint[top:top+h, left:left+width] = 1
            coverage = footprint.reshape(14, 16, 14, 16).mean((1, 3))
            fields = [np.where(coverage > 0, field, np.nan) for field in fields]
            limit = max(float(np.nanmax(np.abs(field))) for field in fields)
            for j, (field, title) in enumerate(zip(fields, [f'class {c}', f'class {e}', f'{c} minus {e}']), 1):
                artist = axes[p, j].imshow(field, cmap='RdBu_r', vmin=-limit, vmax=limit,
                                  interpolation='nearest', extent=[0, 224, 224, 0])
                axes[p, j].set_facecolor('.85')
                axes[p, j].set_title(f'Valid-grid centered evidence: {title}')
                fig.colorbar(artist, ax=axes[p, j], fraction=.046, pad=.02)
                axes[p, j].add_patch(plt.Rectangle((left, top), width, h, fill=False, edgecolor='black'))
        fig.suptitle(f'SOURCE VALIDATION ONLY: {row.sample_id}; true={row.class_id}; B={base[i].argmax()}\n'
                     'Rotation0 display; C4 used in classifier; 14x14 broad receptive fields, NOT physical pixel causes')
        fig.savefig(HERE / 'panels' / f'pair{c}_{e}_class{row.class_id}.png', dpi=100)
        plt.close(fig)
    receipt = dict(rows=len(source), train_rows=len(tr), validation_rows=len(va), test_rows=0,
                   pairs=selected[['c', 'd']].values.tolist(), fits=9,
                   seconds=time.perf_counter()-started, moment_replay_max_error=replay,
                   covariance_identity_max_error=maxerr, source_only=True,
                   candidate_selection_uses_test=False, maximum_inner_iterations=2000,
                   added_features_not_capacity_matched=True)
    (HERE / 'summary.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
