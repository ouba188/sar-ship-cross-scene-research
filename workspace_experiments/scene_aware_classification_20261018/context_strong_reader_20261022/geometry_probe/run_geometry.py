"""Fixed geometry-only diagnostic; cached expanded splits, no reader refitting."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = Path(__file__).resolve().parent
STRONG = HERE.parent
ROOT = STRONG.parent
EXP = ROOT/'context_expansion_20261022'
sys.path.insert(0, str(ROOT/'context_relation_pilot_20261022'))
from run_ports24 import weights, metrics

BODY = ['length_m', 'width_m', 'aspect']
CANVAS = ['canvas_w_vv', 'canvas_h_vv', 'canvas_w_vh', 'canvas_h_vh',
          'occupancy_vv', 'occupancy_vh']
GROUPS = {'body_geometry': BODY, 'canvas_geometry': CANVAS, 'both': BODY + CANVAS}
PARAMS = dict(max_iter=100, max_leaf_nodes=7, l2_regularization=1,
              random_state=17, early_stopping=False)


def summarize(pred):
    rows = []
    for name, q in pred.groupby('model', sort=False):
        rows.append(dict(model=name, n=len(q),
                         **metrics(q.class_id.to_numpy(), q[['s0', 's1', 's2']].to_numpy())))
    return pd.DataFrame(rows)


def geometry_table(t):
    g = t[['instance_id', 'class_id', 'class_name', 'mmsi', 'pass_group']].copy()
    g['length_m'], g['width_m'] = t.target_length_m, t.target_width_m
    g['aspect'] = g.length_m/g.width_m
    for pol in ('vv', 'vh'):
        win = np.array(t['target_window_'+pol].map(json.loads).tolist())
        np.testing.assert_array_equal(win[:, 2:], t[['w_'+pol, 'h_'+pol]].to_numpy())
        g['canvas_w_'+pol], g['canvas_h_'+pol] = win[:, 2], win[:, 3]
        g['occupancy_'+pol] = t['target_occupancy_'+pol]
        np.testing.assert_allclose(g['occupancy_'+pol],
            t['width_'+pol]*t['height_'+pol]/(win[:, 2]*win[:, 3]), atol=1e-12)
    raw = g[BODY + CANVAS].to_numpy(dtype=float)
    assert np.isfinite(raw).all() and (raw > 0).all()
    assert (g[['occupancy_vv', 'occupancy_vh']].to_numpy() <= 1).all()
    for name in BODY + CANVAS:
        g['log_'+name] = np.log(g[name])
    return g


def distributions(g):
    rows = []
    for (c, name), q in g.groupby(['class_id', 'class_name']):
        for field in BODY + CANVAS:
            a = q[field].to_numpy()
            rows.append(dict(class_id=c, class_name=name, variable=field,
                n=len(q), mmsi=q.mmsi.nunique(), mean=float(a.mean()), std=float(a.std()),
                **{key:float(np.quantile(a, value)) for key, value in
                   [('min', 0), ('p05', .05), ('p25', .25), ('median', .5),
                    ('p75', .75), ('p95', .95), ('max', 1)]}))
    pd.DataFrame(rows).to_csv(HERE/'class_size_distributions.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fields = BODY + ['canvas_w_vv', 'occupancy_vv', 'occupancy_vh']
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, field in zip(axes.flat, fields):
        bins = np.linspace(g[field].min(), g[field].max(), 26)
        for (_, name), q in g.groupby(['class_id', 'class_name']):
            ax.hist(q[field], bins=bins, weights=np.ones(len(q))/len(q),
                    histtype='step', linewidth=1.5, label=name)
        ax.set(xlabel=field, ylabel='Within-class fraction')
        ax.grid(alpha=.15)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE/'class_size_distributions.png', dpi=150)
    plt.close(fig)


def comparisons(test, t, g):
    pred = test.pivot(index='instance_id', columns='model', values='prediction')
    pred = pred.reindex(t.instance_id).reset_index()
    out = t[['instance_id', 'class_id', 'class_name', 'mmsi', 'pass_group']].merge(
        pred, on='instance_id', validate='one_to_one')
    out = out.merge(g[['instance_id'] + BODY + CANVAS], on='instance_id', validate='one_to_one')
    out['fold'] = out.instance_id.map(test[test.model.eq('strongbase')].set_index('instance_id').fold)
    y = out.class_id.to_numpy()
    stem = out.stem.eq(y).to_numpy()
    strong = out.strongbase.eq(y).to_numpy()
    out['strong_rescued_vs_stem'] = ~stem & strong
    out['strong_harmed_vs_stem'] = stem & ~strong
    result = []
    for name in GROUPS:
        correct = out[name].eq(y).to_numpy()
        out[name+'_correct'] = correct
        for cohort, use in [('all', np.ones(len(out), bool)),
                            ('strong_rescued_vs_stem', ~stem & strong),
                            ('strong_harmed_vs_stem', stem & ~strong),
                            ('strong_and_stem_correct', stem & strong),
                            ('strong_and_stem_wrong', ~stem & ~strong)]:
            result.append(dict(model=name, cohort=cohort, n=int(use.sum()),
                geometry_correct=int((use & correct).sum()),
                geometry_accuracy=float(correct[use].mean()*100),
                agreement_with_strong=float(out.loc[use, name].eq(out.loc[use, 'strongbase']).mean()*100)))
    out.to_csv(HERE/'per_example_comparison.csv', index=False)
    pd.DataFrame(result).to_csv(HERE/'strong_geometry_overlap.csv', index=False)
    classes, acquisitions, pairs = [], [], []
    for name in ['stem', 'strongbase'] + list(GROUPS):
        correct = out[name].eq(y)
        for c, q in out.groupby('class_id'):
            classes.append(dict(model=name, class_id=c, class_name=q.class_name.iloc[0], n=len(q),
                                correct=int(correct[q.index].sum()), recall=float(correct[q.index].mean()*100)))
        for group, q in out.groupby('pass_group'):
            acquisitions.append(dict(model=name, pass_group=group, n=len(q),
                OA=float(correct[q.index].mean()*100),
                net_vs_strong=int(correct[q.index].sum()-q.strongbase.eq(q.class_id).sum())))
    for name in GROUPS:
        a = out[name].eq(y).to_numpy()
        for ref, b in [('stem', stem), ('strongbase', strong)]:
            pairs.append(dict(model=name, reference=ref, n=len(out),
                both_correct=int((a & b).sum()), geometry_only_correct=int((a & ~b).sum()),
                reference_only_correct=int((~a & b).sum()), both_wrong=int((~a & ~b).sum()),
                delta_OA=float((a.astype(float)-b).mean()*100)))
    pd.DataFrame(classes).to_csv(HERE/'class_metrics.csv', index=False)
    pd.DataFrame(acquisitions).to_csv(HERE/'acquisition_metrics.csv', index=False)
    pd.DataFrame(pairs).to_csv(HERE/'paired_comparison.csv', index=False)
    return out


def case_panels(out, t):
    # Descriptive median-size cases, selected after prediction without changing fits.
    candidates = []
    for c in sorted(out.class_id.unique()):
        q = out[out.class_id.eq(c) & out.strong_rescued_vs_stem & ~out.both_correct].copy()
        candidates.append((f'strong_rescue_geometry_wrong_class{c}', q))
    q = out[out.both_correct & out.strongbase.ne(out.class_id)].copy()
    candidates.append(('geometry_correct_strong_wrong', q))
    records = []
    font = ImageFont.truetype('C:/Windows/Fonts/consola.ttf', 16)
    small = ImageFont.truetype('C:/Windows/Fonts/consola.ttf', 14)
    lookup = t.set_index('instance_id')
    for reason, q in candidates:
        if q.empty:
            continue
        q['median_distance'] = abs(q.length_m-q.length_m.median())
        row = q.sort_values(['median_distance', 'instance_id']).iloc[0]
        r = lookup.loc[row.instance_id]
        panel = Image.new('RGB', (920, 1030), 'white')
        draw = ImageDraw.Draw(panel)
        lines = [f'{row.instance_id} | {row.class_name} | fold {row.fold}', reason,
            f'True={row.class_id} stem={row.stem} strong={row.strongbase} body={row.body_geometry} canvas={row.canvas_geometry} both={row.both}',
            f'OBB L={row.length_m:.1f}m W={row.width_m:.1f}m aspect={row.aspect:.3f}',
            'Native cached pixels, common 3x nearest display; no intensity rescaling.']
        for k, line in enumerate(lines):
            draw.text((12, 10+k*22), line, fill='black', font=font)
        record = dict(row.drop(labels=['median_distance']), selection=reason)
        for col, pol in enumerate(('vv', 'vh')):
            source = EXP/'readout/native_targets'/f'{row.instance_id}_{pol}.png'
            with Image.open(source) as im:
                im.load()
                native = im.convert('RGB')
            win = json.loads(r['target_window_'+pol])
            assert native.size == tuple(win[2:])
            image = native.resize((native.width*3, native.height*3), Image.Resampling.NEAREST)
            x = 10+col*450+(420-image.width)//2
            draw.text((12+col*450, 133),
                f'{pol.upper()} {win[2]}x{win[3]} occ={row["occupancy_"+pol]:.4f}', fill='black', font=font)
            panel.paste(image, (x, 175+(384-image.height)//2))
            panel.paste(image, (x, 610+(384-image.height)//2))
            points = np.asarray(json.loads(r.target_points)) - np.asarray(win[:2])
            offset = np.array([x, 610+(384-image.height)//2])
            polygon = points*3+offset
            draw.line([tuple(p) for p in np.vstack([polygon, polygon[:1]])], fill=(240, 40, 40), width=2)
            draw.text((12+col*450, 580), 'Same pixels + matched VV OBB', fill='black', font=small)
            record['native_'+pol] = str(source)
            record['source_window_'+pol] = r['target_window_'+pol]
        dest = HERE/f'case_{row.instance_id}.png'
        panel.save(dest)
        record['panel'] = str(dest)
        records.append(record)
    pd.DataFrame(records).to_csv(HERE/'review_cases.csv', index=False)
    return [r['instance_id'] for r in records]


def main():
    started = time.perf_counter()
    t = pd.read_csv(EXP/'readout/targets.csv', dtype={'mmsi': str})
    with np.load(EXP/'readout/features.npz') as f:
        np.testing.assert_array_equal(t.instance_id, f['instance_ids'])
    assert t.instance_id.is_unique and t.mmsi.notna().all()
    assert t.groupby('mmsi').class_id.nunique().max() == 1
    g = geometry_table(t)
    g.to_csv(HERE/'geometry_inputs.csv', index=False)
    distributions(g)
    selected = pd.read_csv(STRONG/'selections.csv').set_index('fold')
    source_pred = pd.read_csv(STRONG/'baseline_predictions.csv')
    saved_splits = pd.read_csv(EXP/'splits.csv')
    y, identity = t.class_id.to_numpy(), t.mmsi.to_numpy()
    frames, fold_scores, split_checks, train_weight_rows, replay = [], [], [], [], []
    coverage = np.zeros(len(t), dtype=int)
    for fold in range(6):
        with np.load(EXP/f'fold{fold}_inputs.npz') as d:
            indices = {role:d[role] for role in ('train', 'validation', 'test')}
            stem_score = d['base']
        with np.load(STRONG/f'fold{fold}_base.npz') as d:
            strong_score = d['base']
            for role, ix in indices.items():
                np.testing.assert_array_equal(ix, d[role])
                np.testing.assert_array_equal(t.iloc[ix].instance_id,
                    saved_splits.loc[saved_splits.fold.eq(fold) & saved_splits.role.eq(role), 'instance_id'])
        train, val, test = [indices[k] for k in ('train', 'validation', 'test')]
        coverage[test] += 1
        for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
            assert not set(identity[indices[a]]) & set(identity[indices[b]])
            assert not set(t.iloc[indices[a]].pass_group) & set(t.iloc[indices[b]].pass_group)
        w = weights(y[train], identity[train])
        train_weight_rows.append(pd.DataFrame(dict(fold=fold, instance_id=t.iloc[train].instance_id,
            class_id=y[train], mmsi=identity[train], sample_weight=w)))
        for role, ix in indices.items():
            split_checks.append(dict(fold=fold, role=role, n=len(ix), mmsi=len(set(identity[ix])),
                acquisitions=t.iloc[ix].pass_group.nunique(),
                **{f'class_{c}':int((y[ix] == c).sum()) for c in range(3)}))
        scores = {'stem': stem_score, 'strongbase': strong_score}
        for name, fields in GROUPS.items():
            x = g[['log_'+field for field in fields]].to_numpy()
            clf = HistGradientBoostingClassifier(**PARAMS)
            clf.fit(x[train], y[train], sample_weight=w)
            np.testing.assert_array_equal(clf.classes_, np.arange(3))
            scores[name] = clf.predict_proba(x)
            assert np.isfinite(scores[name]).all()
            np.testing.assert_allclose(scores[name].sum(1), 1, atol=1e-12)
        for name, score in scores.items():
            if name in ('stem', 'strongbase'):
                kind = 'stem' if name == 'stem' else selected.loc[fold, 'kind']
                q = source_pred[source_pred.fold.eq(fold) & source_pred.kind.eq(kind)].set_index('instance_id')
                saved = q.loc[t.iloc[test].instance_id, ['s0', 's1', 's2']].to_numpy()
                # Strong recomputed stem in float32; its own replay tolerance is 3e-4.
                np.testing.assert_allclose(score[test], saved, atol=3e-4 if name == 'stem' else 1e-10)
                np.testing.assert_array_equal(score[test].argmax(1), saved.argmax(1))
                replay.append(dict(fold=fold, model=name,
                                   max_score_error=float(np.abs(score[test]-saved).max()), prediction_changes=0))
            for role in ('validation', 'test'):
                ix = indices[role]
                frame = t.iloc[ix][['instance_id', 'class_id', 'mmsi', 'pass_group']].copy()
                frame['fold'], frame['role'], frame['model'] = fold, role, name
                frame['prediction'] = score[ix].argmax(1)
                frame['score_type'] = 'probability' if name in GROUPS else 'ridge_score'
                for c in range(3):
                    frame['s'+str(c)] = score[ix, c]
                frames.append(frame)
                fold_scores.append(dict(fold=fold, role=role, model=name, n=len(ix),
                    **metrics(y[ix], score[ix])))
        print(f'fold {fold}: train={len(train)} val={len(val)} test={len(test)}; 3 fixed HGB fits complete', flush=True)
    np.testing.assert_array_equal(coverage, np.ones(len(t)))
    pred = pd.concat(frames, ignore_index=True)
    pred.to_csv(HERE/'predictions.csv', index=False)
    pd.DataFrame(fold_scores).to_csv(HERE/'fold_metrics.csv', index=False)
    pd.DataFrame(split_checks).to_csv(HERE/'split_verification.csv', index=False)
    pd.DataFrame(replay).to_csv(HERE/'baseline_replay.csv', index=False)
    pd.concat(train_weight_rows).to_csv(HERE/'training_weights.csv', index=False)
    test = pred[pred.role.eq('test')]
    assert test.groupby('model').instance_id.nunique().eq(len(t)).all()
    pooled = summarize(test)
    pooled.to_csv(HERE/'pooled.csv', index=False)
    out = comparisons(test, t, g)
    cases = case_panels(out, t)
    summary = dict(targets=len(t), identities=t.mmsi.nunique(), acquisitions=t.pass_group.nunique(),
        folds=6, fits=18, parameters=PARAMS, inputs={k:['log_'+x for x in v] for k,v in GROUPS.items()},
        occupancy='Frozen original target bounding-box area divided by native canvas area, independently VV/VH; not readout OBB-mask fraction.',
        body='Matched valid VV detection OBB lengths in metres; not AIS hull measurements.',
        weights='Imported unchanged run_ports24.weights; class balance then equal MMSI within class, mean-one.',
        validation_usage='Report only; no tuning, early stopping, model/group selection, or refit on validation.',
        checks=dict(ids_aligned=True, folds_match_expanded_and_strong=True, mmsi_and_acquisition_disjoint=True,
                    each_test_once=True, baseline_score_replay=True, finite_inputs_and_outputs=True),
        strong_rescues=int(out.strong_rescued_vs_stem.sum()), strong_harms=int(out.strong_harmed_vs_stem.sum()),
        cases=cases, seconds=time.perf_counter()-started,
        context_boundary='Strong-reader external context has slightly lower OA but slightly positive BA; not uniform degradation.')
    (HERE/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(pooled.to_string(index=False), flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
