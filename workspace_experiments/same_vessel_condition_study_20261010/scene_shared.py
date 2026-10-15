"""Cross-class scene residual diagnostics; fixed heads, not a deployment model."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE/'scene_shared_nuisance_20261010'
SEED = 20261010
MIN_DONORS = 3
MAX_DONORS = 8
MIN_OTHER_CLASSES = 2
MODES = ('r50', 'r18', 'dual')
LEVELS = ('global_class_reference', 'within_port_class_reference')


def balanced_offsets(same, other, references):
    return np.mean(same-references, axis=0), np.mean(other-references, axis=0)


def corrected_scores(scores, offset, coef, strength):
    return scores-strength*(offset@coef.T)


def transform_all(raw, model, name):
    rank = model[name+'_components'].shape[0]
    result = np.empty((len(ROWS), rank), np.float32)
    offset = model[name+'_pca_mean']@model[name+'_components'].T
    for start in range(0, len(ROWS), 128):
        sl = slice(start, min(start+128, len(ROWS)))
        block = (np.asarray(raw[ROWS[sl]], np.float32)-model[name+'_center'])/model[name+'_scale']
        result[sl] = (block@model[name+'_components'].T-offset)/np.sqrt(rank)
    return result


def view(x, mode):
    if mode == 'r50':
        return x[..., :128]
    if mode == 'r18':
        return x[..., 128:]
    return x/np.sqrt(2.)


def cosine(x, y):
    return float(x@y/max(np.linalg.norm(x)*np.linalg.norm(y), 1e-12))


def choose_control(scene, query_class, donor_classes, cells, port_scenes, scene_times):
    candidates = sorted((s for s in port_scenes if s != scene),
                        key=lambda s: (abs(scene_times[s]-scene_times[scene]), s))
    for other in candidates:
        if not all((other, c) in cells for c in donor_classes):
            continue
        refs = {c: [s for s in port_scenes if s not in (scene, other) and (s, c) in cells]
                for c in (*donor_classes, query_class)}
        if all(len(v) >= 2 for v in refs.values()):
            return other, refs
    return None, None


def build_cells(obs, fold):
    train = obs.loc[obs.fold.ne(fold)]
    cells = {key: g.obs_id.to_numpy() for key, g in train.groupby(['product', 'class_id'])
             if g.mmsi.nunique() >= MIN_DONORS}
    return cells


def run_fold(obs, fold, raw, expected):
    started = time.perf_counter()
    with np.load(BASE/f'fold_{fold}_models.npz') as z:
        model = {k: z[k] for k in z.files}
    x = np.empty((len(obs), 192), np.float32)
    for name, sl in [('r50', slice(0, 128)), ('r18', slice(128, 192))]:
        x[:, sl] = transform_all(raw[name], model, name)
    query_index = np.flatnonzero(obs.fold.to_numpy() == fold)
    errors, bases, spans = {}, {}, {}
    for mi, mode in enumerate(MODES):
        coef = model[mode+'_head_coef']
        score = view(x[query_index], mode)@coef.T+model[mode+'_head_intercept']
        np.testing.assert_allclose(score, expected[query_index, mi], atol=1e-5, rtol=2e-5)
        np.testing.assert_array_equal(score.argmax(1), expected[query_index, mi].argmax(1))
        errors[mode] = float(np.max(abs(score-expected[query_index, mi])))
        bases[mode] = pd.DataFrame(score, index=query_index)
        _, singular, vectors = np.linalg.svd(coef-coef.mean(0), full_matrices=False)
        spans[mode] = vectors[singular > singular[0]*1e-7]
    cells = build_cells(obs, fold)
    centroids = {k: x[ix].mean(0) for k, ix in cells.items()}
    scene_info = obs[['product', 'port', 'acq_time']].drop_duplicates('product').set_index('product')
    port_scenes = {p: g.index.tolist() for p, g in scene_info.groupby('port')}
    scene_times = pd.to_datetime(scene_info.acq_time).astype('int64').to_dict()
    class_scenes = {c: [s for s, cc in cells if cc == c] for c in range(8)}
    records, geometry, matching, examples = [], [], [], []
    vectors_same, vectors_other, vectors_query = [], [], []
    coverage = dict(fold=fold, query_cells=0, matched_cells=0, skipped_other_classes=0,
                    skipped_control_or_reference=0)
    target = obs.loc[obs.fold.eq(fold)]
    for cell_id, ((scene, cls), q) in enumerate(target.groupby(['product', 'class_id'])):
        coverage['query_cells'] += 1
        donor_classes = [c for c in range(8) if c != cls and (scene, c) in cells]
        if len(donor_classes) < MIN_OTHER_CLASSES:
            coverage['skipped_other_classes'] += 1
            continue
        port = q.port.iloc[0]
        other, local_refs = choose_control(scene, cls, donor_classes, cells,
                                           port_scenes[port], scene_times)
        if other is None:
            coverage['skipped_control_or_reference'] += 1
            continue
        coverage['matched_cells'] += 1
        rng = np.random.default_rng(SEED+fold*100000+cell_id)
        a, b, source_rows, other_rows, support_counts = [], [], [], [], []
        for c in donor_classes:
            n = min(MAX_DONORS, len(cells[scene, c]), len(cells[other, c]))
            ia = rng.choice(cells[scene, c], n, replace=False)
            ib = rng.choice(cells[other, c], n, replace=False)
            a.append(x[ia].mean(0)); b.append(x[ib].mean(0))
            source_rows.extend(ia.tolist()); other_rows.extend(ib.tolist()); support_counts.append(n)
        a, b = np.stack(a), np.stack(b)
        assert not set(q.mmsi) & set(obs.iloc[source_rows+other_rows].mmsi)
        matching.append(dict(fold=fold, product=scene, control_product=other, port=port,
            class_id=cls, queries=len(q), donor_classes=','.join(map(str, donor_classes)),
            supports_per_class=','.join(map(str, support_counts)),
            donor_obs_ids=','.join(map(str, source_rows)), control_obs_ids=','.join(map(str, other_rows)),
            query_obs_ids=','.join(map(str, q.obs_id)),
            control_time_gap_days=abs(scene_times[scene]-scene_times[other])/86400e9))
        for level in LEVELS:
            ref_scenes = local_refs if level == LEVELS[1] else {
                c: [s for s in class_scenes[c] if s not in (scene, other)]
                for c in (*donor_classes, cls)}
            ref = {c: np.mean([centroids[s, c] for s in ss], axis=0) for c, ss in ref_scenes.items()}
            same, control = balanced_offsets(a, b, np.stack([ref[c] for c in donor_classes]))
            residual = x[q.obs_id].mean(0)-ref[cls]
            vector_id = len(vectors_same)
            vectors_same.append(same); vectors_other.append(control); vectors_query.append(residual)
            for mode in MODES:
                bs, bc, qr = [view(v, mode) for v in (same, control, residual)]
                w = model[mode+'_head_coef']
                score = bases[mode].loc[q.obs_id].to_numpy()
                before = score.argmax(1)
                rival_score = score.copy(); rival_score[:, cls] = -np.inf
                rival = rival_score.argmax(1)
                ss, sc = bs@w.T, bc@w.T
                span = spans[mode]
                geometry.append(dict(fold=fold, level=level, product=scene, control_product=other,
                    port=port, class_id=cls, mode=mode, queries=len(q), vector_id=vector_id,
                    same_cosine=cosine(bs, qr), control_cosine=cosine(bc, qr),
                    cosine_advantage=cosine(bs, qr)-cosine(bc, qr),
                    same_residual_mse=float(np.mean((qr-bs)**2)),
                    control_residual_mse=float(np.mean((qr-bc)**2)),
                    same_decision_energy_fraction=float(np.sum((bs@span.T)**2)/max(bs@bs, 1e-12)),
                    same_margin_damage=float(np.mean(ss[rival]-ss[cls])),
                    control_margin_damage=float(np.mean(sc[rival]-sc[cls]))))
                for strength in (.5, 1.):
                    for arm, offset in [('same_scene', bs), ('other_scene', bc)]:
                        after_score = corrected_scores(score, offset, w, strength)
                        after = after_score.argmax(1)
                        ok0, ok1 = before == cls, after == cls
                        rescue, harm = ~ok0 & ok1, ok0 & ~ok1
                        records.append(dict(fold=fold, level=level, product=scene, port=port,
                            class_id=cls, mode=mode, arm=arm, strength=strength, n=len(q),
                            correct_base=int(ok0.sum()), correct_after=int(ok1.sum()),
                            rescues=int(rescue.sum()), harms=int(harm.sum())))
                        if level == LEVELS[1] and arm == 'same_scene' and strength == .5:
                            for outcome, mask in [('rescue', rescue), ('harm', harm)]:
                                choices = np.flatnonzero(mask)
                                if len(choices):
                                    k = choices[0]
                                    examples.append(dict(fold=fold, product=scene, port=port,
                                        class_id=cls, mode=mode, outcome=outcome,
                                        obs_id=int(q.obs_id.iloc[k]), baseline=int(before[k]),
                                        corrected=int(after[k]), competing_class=int(rival[k]),
                                        margin_before=float(score[k, cls]-score[k, rival[k]]),
                                        margin_after=float(after_score[k, cls]-after_score[k, rival[k]])))
    for name, data in [('outcomes', records), ('geometry', geometry), ('matches', matching), ('cases', examples)]:
        pd.DataFrame(data).to_csv(OUT/f'fold_{fold}_{name}.csv', index=False)
    np.savez_compressed(OUT/f'fold_{fold}_residual_vectors.npz', same=np.stack(vectors_same),
                        control=np.stack(vectors_other), query=np.stack(vectors_query))
    coverage.update(replay_errors=errors, predictions_exact=True, seconds=time.perf_counter()-started)
    (OUT/f'fold_{fold}_receipt.json').write_text(json.dumps(coverage, indent=2), encoding='utf-8')
    print(json.dumps(coverage), flush=True)


def summarize():
    outcomes = pd.concat([pd.read_csv(OUT/f'fold_{f}_outcomes.csv') for f in range(3)], ignore_index=True)
    geometry = pd.concat([pd.read_csv(OUT/f'fold_{f}_geometry.csv') for f in range(3)], ignore_index=True)
    groups = ['level', 'mode', 'arm', 'strength']
    class_rows, total_rows = [], []
    for key, g in outcomes.groupby(groups):
        cls = g.groupby('class_id')[['n', 'correct_base', 'correct_after', 'rescues', 'harms']].sum()
        rates = cls[['correct_base', 'correct_after', 'rescues', 'harms']].div(cls.n, axis=0)*100
        for c in cls.index:
            class_rows.append(dict(zip(groups, key), class_id=c, queries=int(cls.loc[c, 'n']),
                scenes=g.loc[g.class_id.eq(c), 'product'].nunique(), **rates.loc[c].to_dict(),
                delta=rates.loc[c, 'correct_after']-rates.loc[c, 'correct_base']))
        sums = cls.sum()
        total_rows.append(dict(zip(groups, key), queries=int(sums.n), scenes=g['product'].nunique(),
            ports=g.port.nunique(), classes=len(cls), OA_base=100*sums.correct_base/sums.n,
            OA_after=100*sums.correct_after/sums.n, BA_base=rates.correct_base.mean(),
            BA_after=rates.correct_after.mean(), delta_BA=(rates.correct_after-rates.correct_base).mean(),
            rescue_BA=rates.rescues.mean(), harm_BA=rates.harms.mean()))
    summary = pd.DataFrame(total_rows)
    summary.to_csv(OUT/'classification_summary.csv', index=False)
    pd.DataFrame(class_rows).to_csv(OUT/'classification_by_class.csv', index=False)
    geo_fields = ['same_cosine', 'control_cosine', 'cosine_advantage', 'same_residual_mse',
                  'control_residual_mse', 'same_decision_energy_fraction',
                  'same_margin_damage', 'control_margin_damage']
    # Each scene-class gets one value after averaging its three identity-fold diagnostics.
    by_scene_class = geometry.groupby(['level', 'mode', 'class_id', 'product'])[geo_fields].mean()
    by_class = by_scene_class.groupby(['level', 'mode', 'class_id']).mean()
    by_class.to_csv(OUT/'geometry_by_class.csv')
    by_class.groupby(['level', 'mode']).mean().to_csv(OUT/'geometry_summary.csv')
    outcomes.groupby(groups+['port'])[['n', 'correct_base', 'correct_after', 'rescues', 'harms']].sum().to_csv(OUT/'classification_by_port.csv')
    matches = pd.concat([pd.read_csv(OUT/f'fold_{f}_matches.csv') for f in range(3)], ignore_index=True)
    eligible = np.unique(np.concatenate([np.fromstring(s, sep=',', dtype=int) for s in matches.query_obs_ids]))
    obs = pd.read_csv(BASE/'observations.csv')
    receipt = dict(query_observations=len(eligible), query_ships=int(obs.iloc[eligible].mmsi.nunique()),
                   scenes=int(matches['product'].nunique()), ports=int(matches.port.nunique()),
                   scene_class_fold_cells=len(matches), complete=True,
                   warning='Label-conditioned mechanistic diagnostic, NOT deployable target accuracy improvement')
    (OUT/'COMPLETE.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(summary[['level', 'mode', 'arm', 'strength', 'queries', 'scenes', 'delta_BA']].round(4).to_string(index=False), flush=True)
    print(json.dumps(receipt), flush=True)


def run():
    global ROWS
    OUT.mkdir(exist_ok=True)
    protocol = dict(seed=SEED, date='2026-10-10', hypothesis='Shared scene factors may affect class decisions; class/domain interactions are allowed',
        source='24-port existing frozen identity-fold R50/R18 diagnostic heads; not main LOPO',
        scene_key='product', identity='MMSI proxy, existing disjoint three folds',
        reference='Equal-scene class centroids; both compared scenes excluded; at least two reference scenes',
        donor='Other classes only, >=2 classes, >=3 distinct source-fold ships per class, max8 per class',
        control='Closest-time different scene in SAME port, same donor classes and sample counts',
        levels=list(LEVELS), correction='Subtract residual from frozen PCA features; strengths0.5 primary,1 secondary, no selection',
        label_boundary='Query true class defines excluded donor class and evaluation cohort. Oracle diagnostic; source-fold donor labels exceed main64-label setting.',
        no_query_identities_in_donors_or_references=True,
        primary='Cross-class cosine advantage over matched other-scene control, and correction rescue/harm versus control',
        limitations=['Shared scene is not pure noise or identified physical cause',
                     'Existing heads and frozen encoders have historical scene/domain exposure',
                     'References retain subclass and unobserved ship-state differences',
                     'Nearest other-scene comparison not a randomized intervention',
                     'Does not quantify the actual within-port vs LOPO gap'],
        resources='CPU1, batch128, no GPU or parent-project modifications')
    (OUT/'PROTOCOL.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    obs = pd.read_csv(BASE/'observations.csv')
    assert obs.groupby('mmsi').fold.nunique().max() == 1
    assert obs.groupby('product').port.nunique().max() == 1
    assert not obs.duplicated(['product', 'mmsi']).any()
    ROWS = obs.manifest_row.to_numpy()
    raw = dict(r50=np.load('E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/pooled_features.npy', mmap_mode='r'),
               r18=np.load('E:/Hermes/paper_tracking/sensitivity/neural_complement_pilot_20261007/frozen_replay/source_extra.npy', mmap_mode='r'))
    with np.load(BASE/'FROZEN_PREDICTIONS.npz') as z:
        expected = z['scores']
        np.testing.assert_array_equal(z['manifest_rows'], ROWS)
    for fold in range(3):
        run_fold(obs, fold, raw, expected)
    summarize()


if __name__ == '__main__':
    run()
