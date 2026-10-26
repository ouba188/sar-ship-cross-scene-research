"""Read frozen M3 source predictions; write held-acquisition evaluation only here."""
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
os.environ['MPLCONFIGDIR'] = str(HERE / '.mplcache')
os.environ['PYTHONIOENCODING'] = 'utf-8'
import argparse
import ast
import importlib.util
import itertools
import json
import numpy as np
import pandas as pd

SAVED = ROOT / 'new_version_ablations_20261026/m3'
spec = importlib.util.spec_from_file_location('cached_m3_runner', SAVED / 'run.py')
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)
tr, m3 = run.tr, run.m3
SEEDS, ARMS, ENDPOINTS = run.SEEDS, run.ARMS, run.ENDPOINTS
CLASSES = ('bulk', 'general', 'chemical', 'container')
DRAW_SEED, DRAWS = 1026, 4000


def write(name, obj):
    (HERE / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def save(name, rows):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(HERE / name, index=False, encoding='utf-8-sig')
    return frame


def aggregate(frame, keys, values):
    out = frame.groupby(keys, dropna=False)[values].agg(['mean', 'std']).reset_index()
    out.columns = ['_'.join(x for x in col if x) for col in out.columns]
    return out


def load_source(source):
    target = 'oss2' if source == 'oss1' else 'oss1'
    direction = f'{source}-to-{target}'
    table = pd.read_csv(ROOT / f'{source}_m123_replication_20261026/stack_table.csv', dtype={'mmsi': str})
    ids = np.flatnonzero(table.role.eq('evaluation'))
    held = table.iloc[ids].copy().reset_index(drop=True)
    held['source_row'] = ids
    assert len(held) == {'oss1': 63, 'oss2': 378}[source]
    assert table.obs_id.is_unique
    supervised = table[table.role.ne('evaluation')]
    for key in ('mmsi', 'pass_group'):
        assert not set(held[key]) & set(supervised[key]), (source, key, 'held-role overlap')
    data = {}
    for arm, seed in itertools.product(ARMS, SEEDS):
        path = SAVED / direction / f'seed{seed}' / arm / 'source_predictions.npz'
        z = tr.load_npz(path)
        np.testing.assert_array_equal(z['obs_id'], table.obs_id)
        data[arm, seed] = {k: v[ids] for k, v in z.items()}
    anchor = data['both', SEEDS[0]]['refs']
    original_refs = tr.load_npz(ROOT / f'{source}_m123_replication_20261026/m3_results/references.npz')
    np.testing.assert_array_equal(anchor, original_refs['refs'][ids])
    assert np.isin(anchor, original_refs['bank']).all()
    rr = table.iloc[anchor.ravel()]
    assert rr.role.eq('fit_train').all()
    np.testing.assert_array_equal(rr.class_id.to_numpy().reshape(len(ids), 4, 2), np.broadcast_to(np.arange(4)[None, :, None], anchor.shape))
    for key in ('mmsi', 'pass_group'):
        assert not (held[key].to_numpy()[:, None, None] == table[key].to_numpy()[anchor]).any()
    for arm, seed in itertools.product(ARMS, SEEDS):
        z = data[arm, seed]
        np.testing.assert_array_equal(z['refs'], anchor)
        np.testing.assert_array_equal(z['bare'], data['both', seed]['bare'])
        for endpoint in ('B_fixed', 'spatial', 'frequency'):
            reconstructed = z['bare'] + z[f'weight_{endpoint}'][:, :1]*z['delta_vv'] + z[f'weight_{endpoint}'][:, 1:]*z['delta_vh']
            np.testing.assert_array_equal(z[endpoint], reconstructed)
        retained = tr.load_npz(run.ALIGNED / direction / f'seed{seed}/source_predictions.npz')
        np.testing.assert_array_equal(retained['obs_id'], table.obs_id)
        if arm == 'both':
            for endpoint in ('bare', 'B_fixed', 'frequency'):
                np.testing.assert_array_equal(z[endpoint], retained[endpoint][ids])
    return table, held, data, direction


def paired(y, before, after):
    bp, ap = before.argmax(1), after.argmax(1)
    b, a = bp == y, ap == y
    d = a.astype(float) - b
    return dict(n=len(y), rescue=int((a & ~b).sum()), harm=int((b & ~a).sum()),
                CC=int((a & b).sum()), WW=int((~a & ~b).sum()),
                WW_prediction_flips=int(((~a & ~b) & (bp != ap)).sum()),
                OA_pp=100*d.mean(), BA_pp=100*np.mean([d[y == c].mean() for c in np.unique(y)]))


def uncertainty(table, effects, group_key):
    y = table.class_id.to_numpy(int)
    d = np.mean(effects, axis=0)
    groups = list(table.groupby(group_key).indices.values())
    result = dict(group_key=group_key, groups=len(groups), draws=DRAWS,
                  OA_pp=100*d.mean(), BA_pp=100*np.mean([d[y == c].mean() for c in range(4)]),
                  seed_averaging='within observation before paired resampling')
    if len(groups) == 1:
        return dict(**result, OA_low=np.nan, OA_high=np.nan, BA_low=np.nan, BA_high=np.nan,
                    BA_complete_draws=0, scope='single acquisition: cross-acquisition CI unavailable')
    count = np.array([np.bincount(y[ix], minlength=4) for ix in groups])
    total = np.array([np.bincount(y[ix], weights=d[ix], minlength=4) for ix in groups])
    ix = np.random.default_rng(DRAW_SEED).integers(len(groups), size=(DRAWS, len(groups)))
    count, total = count[ix].sum(1), total[ix].sum(1)
    complete = (count > 0).all(1)
    oa = 100*total.sum(1)/count.sum(1)
    ba = 100*(total[complete]/count[complete]).mean(1)
    return dict(**result, OA_low=float(np.quantile(oa, .025)), OA_high=float(np.quantile(oa, .975)),
                BA_low=float(np.quantile(ba, .025)), BA_high=float(np.quantile(ba, .975)),
                BA_complete_draws=int(complete.sum()),
                scope=('few-acquisition cluster bootstrap; exploratory' if group_key == 'pass_group'
                       else 'ship cluster bootstrap conditional on these observed acquisitions; not cross-pass CI'))


def comparisons():
    result = []
    for endpoint in ('B_fixed', 'spatial', 'frequency'):
        result.extend((a, endpoint, b, endpoint) for a, b in itertools.combinations(ARMS, 2))
    for arm in ARMS:
        result.extend((arm, a, arm, b) for a, b in
                      (('bare', 'B_fixed'), ('bare', 'frequency'), ('B_fixed', 'spatial'),
                       ('B_fixed', 'frequency'), ('spatial', 'frequency')))
    return result


def evaluate():
    rows, classrows, passmetrics, pairs, byclass, bypass, bypassclass = [], [], [], [], [], [], []
    ci, events, gates, operators, versions, coverage, refrows, checks = [], [], [], [], [], [], [], []
    selection = pd.read_csv(SAVED / 'selections.csv')
    historical = pd.read_csv(SAVED / 'metrics.csv')
    for source in ('oss1', 'oss2'):
        table, held, data, direction = load_source(source)
        risk_metric_max_error = 0.0
        y = held.class_id.to_numpy(int)
        save(f'{source}_evaluation_cohort.csv', held)
        views = pd.read_csv(ROOT / f'{source}_m123_replication_20261026/view_inventory.csv').set_index(['obs_id', 'pol'])
        refs = data['both', SEEDS[0]]['refs']
        for i, query in held.iterrows():
            for c, slot in itertools.product(range(4), range(2)):
                row = int(refs[i, c, slot])
                ref = table.iloc[row]
                refrows.append(dict(source=source, query_row=int(query.source_row), query_obs_id=int(query.obs_id),
                    query_sample_id=query.sample_id, query_mmsi=query.mmsi, query_pass=query.pass_group,
                    reference_class=c, reference_slot=slot, reference_row=row, reference_obs_id=int(ref.obs_id),
                    reference_sample_id=ref.sample_id, reference_mmsi=ref.mmsi, reference_pass=ref.pass_group,
                    reference_role=ref.role, native_vv=views.loc[(ref.obs_id, 'vv'), 'path'],
                    native_vh=views.loc[(ref.obs_id, 'vh'), 'path']))
        for (pg, c), group in held.groupby(['pass_group', 'class_id']):
            coverage.append(dict(source=source, pass_group=pg, class_id=c, class_name=CLASSES[c], n=len(group), ships=group.mmsi.nunique()))
        for arm, seed in itertools.product(ARMS, SEEDS):
            z = data[arm, seed]
            receipt = selection.query('source == @source and seed == @seed and arm == @arm').iloc[0].to_dict()
            for endpoint in ENDPOINTS:
                metric = m3.metrics(y, z[endpoint])
                rows.append(dict(source=source, arm=arm, seed=seed, endpoint=endpoint, **metric))
                old = historical.query('source == @source and side == "source" and role == "evaluation" and arm == @arm and seed == @seed and endpoint == @endpoint').iloc[0]
                np.testing.assert_allclose([metric[k] for k in ('OA', 'BA', 'J')], old[['OA', 'BA', 'J']].to_numpy(float), atol=1e-12, rtol=0)
                risk_error = abs(np.array([metric[k] for k in ('NLL', 'Brier')]) - old[['NLL', 'Brier']].to_numpy(float)).max()
                risk_metric_max_error = max(risk_metric_max_error, float(risk_error))
                assert risk_error < 2e-6, (source, arm, seed, endpoint, risk_error)
                for c in range(4):
                    ids = y == c
                    classrows.append(dict(source=source, arm=arm, seed=seed, endpoint=endpoint, class_id=c,
                                          class_name=CLASSES[c], n=int(ids.sum()), recall=float((z[endpoint].argmax(1)[ids] == c).mean())))
                for pg, ix in held.groupby('pass_group').indices.items():
                    passmetrics.append(dict(source=source, arm=arm, seed=seed, endpoint=endpoint, pass_group=pg,
                                            classes_present=len(np.unique(y[ix])), **m3.metrics(y[ix], z[endpoint][ix])))
            for gate in ('spatial', 'frequency'):
                wd = abs(z[f'weight_{gate}'] - z['weight_B_fixed'])
                sd = abs(z[gate] - z['B_fixed'])
                gates.append(dict(source=source, arm=arm, seed=seed, gate=gate, selected_step=int(receipt[f'{gate}_step']),
                    n=len(held), weight_changed_n=int((wd > 0).any(1).sum()), weight_max=float(wd.max()),
                    score_changed_n=int((sd > 0).any(1).sum()), score_max=float(sd.max()),
                    prediction_changed_n=int((z[gate].argmax(1) != z['B_fixed'].argmax(1)).sum()),
                    equals_B_fixed=bool(np.array_equal(z[gate], z['B_fixed']))))
            operators.append(dict(source=source, arm=arm, seed=seed, metric_step=int(receipt['metric_step']),
                metric_parameter_change=receipt['metric_parameter_change'], vv_step=int(receipt['vv_step']),
                vh_step=int(receipt['vh_step']), alpha=receipt['alpha'],
                local_max_vs_both=float(abs(z['local'] - data['both', seed]['local']).max()),
                final_max_vs_both=float(abs(z['frequency'] - data['both', seed]['frequency']).max()),
                final_max_vs_bare=float(abs(z['frequency'] - z['bare']).max())))
        versions_data = {}
        for seed in SEEDS:
            oldpath = run.aligned.OLD / direction / f'seed{seed}/source_predictions.npz'
            old = tr.load_npz(oldpath)
            np.testing.assert_array_equal(old['obs_id'], table.obs_id)
            versions_data[seed] = old['frequency'][held.source_row]
            np.testing.assert_array_equal(old['bare'][held.source_row], data['both', seed]['bare'])
            for name, score, path in (
                ('old_raw_summary_pre_alignment', versions_data[seed], oldpath),
                ('new_retained_aligned_both', data['both', seed]['frequency'], SAVED/direction/f'seed{seed}/both/source_predictions.npz')):
                versions.append(dict(source=source, seed=seed, version=name, endpoint='frequency', prediction_file=str(path), **m3.metrics(y, score)))
        allpairs = comparisons() + [('old_raw_summary_pre_alignment', 'frequency', 'both', 'frequency')]
        for ba, be, aa, ae in allpairs:
            meta = dict(source=source, before_arm=ba, before=be, after_arm=aa, after=ae)
            effects = []
            for seed in SEEDS:
                b = versions_data[seed] if ba == 'old_raw_summary_pre_alignment' else data[ba, seed][be]
                a = data[aa, seed][ae]
                pair = paired(y, b, a)
                pairs.append(dict(**meta, seed=seed, **pair))
                d = (a.argmax(1) == y).astype(float) - (b.argmax(1) == y)
                effects.append(d)
                for c in range(4):
                    ix = y == c
                    byclass.append(dict(**meta, seed=seed, class_id=c, class_name=CLASSES[c], **paired(y[ix], b[ix], a[ix])))
                for pg, ix in held.groupby('pass_group').indices.items():
                    bypass.append(dict(**meta, seed=seed, pass_group=pg, classes_present=len(np.unique(y[ix])), **paired(y[ix], b[ix], a[ix])))
                    for c in np.unique(y[ix]):
                        sub = ix[y[ix] == c]
                        bypassclass.append(dict(**meta, seed=seed, pass_group=pg, class_id=c, **paired(y[sub], b[sub], a[sub])))
                bc, ac = b.argmax(1) == y, a.argmax(1) == y
                for i, query in held.iterrows():
                    rival = int(np.where(np.arange(4) == y[i], -np.inf, a[i] if not ac[i] else b[i]).argmax())
                    events.append(dict(**meta, seed=seed, source_row=int(query.source_row), obs_id=int(query.obs_id),
                        sample_id=query.sample_id, mmsi=query.mmsi, pass_group=query.pass_group, class_id=int(y[i]),
                        before_pred=int(b[i].argmax()), after_pred=int(a[i].argmax()),
                        outcome='CC' if bc[i] and ac[i] else 'WW' if not bc[i] and not ac[i] else 'rescue' if ac[i] else 'harm',
                        fixed_rival=rival, before_margin=float(b[i, y[i]]-b[i, rival]), after_margin=float(a[i, y[i]]-a[i, rival])))
            ci.extend(dict(**meta, **uncertainty(held, effects, key)) for key in ('pass_group', 'mmsi'))
        checks.append(dict(source=source, evaluation_n=len(held), ships=held.mmsi.nunique(), passes=held.pass_group.nunique(),
            full_source_n=len(table), excluded_roles=table.loc[table.role.ne('evaluation'), 'role'].value_counts().to_dict(),
            original_obs_ids_exact=True, retained_both_exact=True, cached_OA_BA_J_exact=True,
            cached_risk_metric_max_error=risk_metric_max_error,
            common_references_exact=True, references_fit_only=True, query_reference_ship_pass_disjoint=True,
            held_roles_ship_pass_disjoint=True, polar_score_reconstruction_exact=True))
    metrics = save('metrics.csv', rows)
    save('summary.csv', aggregate(metrics, ['source', 'arm', 'endpoint'], ['OA', 'BA', 'J', 'NLL', 'Brier']))
    classes = save('per_class.csv', classrows)
    save('per_class_summary.csv', aggregate(classes, ['source', 'arm', 'endpoint', 'class_id', 'class_name'], ['recall']))
    save('per_pass_metrics.csv', passmetrics)
    paired_frame = save('paired.csv', pairs)
    pkeys = ['source', 'before_arm', 'before', 'after_arm', 'after']
    save('paired_summary.csv', aggregate(paired_frame, pkeys, ['OA_pp', 'BA_pp', 'rescue', 'harm', 'CC', 'WW']))
    save('paired_per_class.csv', byclass)
    save('paired_per_pass.csv', bypass)
    save('paired_per_pass_class.csv', bypassclass)
    save('uncertainty.csv', ci)
    save('case_index.csv', events)
    save('gate_activity.csv', gates)
    save('operator_activity.csv', operators)
    ver = save('version_metrics.csv', versions)
    save('version_summary.csv', aggregate(ver, ['source', 'version', 'endpoint'], ['OA', 'BA', 'J', 'NLL', 'Brier']))
    save('coverage.csv', coverage)
    save('reference_map.csv', refrows)
    write('checks.json', checks)
    write('PROTOCOL_M3.json', dict(date='2026-10-26', scope='within-dataset held-acquisition, source role evaluation only',
        seeds=list(SEEDS), arms=list(ARMS), endpoints=list(ENDPOINTS), cached_root=str(SAVED), new_fits=0,
        source_table='ROOT/{oss1,oss2}_m123_replication_20261026/stack_table.csv',
        id_rule='source_predictions.obs_id exactly equals stack_table.obs_id; refs index full source rows, NOT obs IDs',
        bank='32 source fit_train ships, 8/class; actual 2/class retrieved references/query, identical across arms/seeds',
        uncertainty=dict(draws=DRAWS, seed=DRAW_SEED, seed_std='sample standard deviation, ddof=1',
            acquisition='paired acquisition clusters, mean effects across seeds first; OSS1 no cross-pass interval; OSS2 five clusters exploratory',
            ships='paired MMSI clusters conditional on the observed acquisitions; not independent acquisition uncertainty',
            BA='macro over the four fixed classes; bootstrap BA drops incomplete-class draws and records count'),
        version_boundary='old raw_summary_pre_alignment -> retained aligned both. Neither is the original formal M1/M2 score chain.',
        reference_controls='same raw_summary base; four operators use their previously fitted dependent polar heads and gates',
        existing_development_exposure=True, evaluation_reuse='existing held cohort reanalysis, not a new untouched external confirmation',
        scope_limit='No full-source role metrics enter efficacy. No target-side observations enter this evaluation. No Git or model/default edits.'))
    print(metrics.query('endpoint == "frequency"').groupby(['source', 'arm'])[['OA', 'BA']].agg(['mean', 'std']).to_string(), flush=True)


def draw_cases():
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    # Reuse the existing drawing function without executing its module-level
    # MPLCONFIGDIR assignment, which points outside this task's output directory.
    helper = ROOT / 'new_version_ablations_20261026/analyze_readout.py'
    node = next(n for n in ast.parse(helper.read_text(encoding='utf-8')).body if isinstance(n, ast.FunctionDef) and n.name == 'display')
    scope = {'tr': tr}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(helper), 'exec'), scope)
    display = scope['display']
    index = pd.read_csv(HERE / 'case_index.csv')
    panels = HERE / 'panels'
    panels.mkdir(exist_ok=True)
    cases, absent = [], []
    for source in ('oss1', 'oss2'):
        table, held, data, direction = load_source(source)
        tr.extract.VIEW_TABLE = ROOT / f'{source}_m123_replication_20261026/view_inventory.csv'
        geo = ROOT / f'{source}_m123_replication_20261026/geometry.csv'
        tr.extract.RADAR_TABLE = geo if geo.exists() else ROOT.parents[1] / 'oss1_joint_geometry_20261017/geometry.csv'
        records = tr.extract.prepare_inventory(table)
        chosen = index.query('source == @source and before_arm == "both" and after_arm == "cosine_fixed" and before == "frequency" and after == "frequency"')
        for outcome in ('rescue', 'harm', 'CC', 'WW'):
            group = chosen[chosen.outcome.eq(outcome)].sort_values(['seed', 'obs_id'])
            if group.empty:
                absent.append(dict(source=source, comparison='both frequency -> cosine_fixed frequency', outcome=outcome))
                continue
            item = group.iloc[0]
            seed, obs = int(item.seed), int(item.obs_id)
            j = int(np.flatnonzero(held.obs_id.to_numpy() == obs)[0])
            row, true, rival = int(held.source_row.iloc[j]), int(item.class_id), int(item.fixed_rival)
            b, a = data['both', seed], data['cosine_fixed', seed]
            refs = a['refs'][j, [true, rival]].ravel()
            entry = item.to_dict()
            entry['selection_rule'] = 'first seed in 17/29/43 with this state, then smallest original obs_id; not largest effect'
            entry['base_margin'] = float(a['bare'][j, true] - a['bare'][j, rival])
            for name, z in (('before', b), ('after', a)):
                weights = z['weight_frequency'][j]
                dp = np.array([z['delta_vv'][j, true]-z['delta_vv'][j, rival], z['delta_vh'][j, true]-z['delta_vh'][j, rival]])
                contribution = weights * dp
                entry[f'{name}_weights'] = weights.tolist()
                entry[f'{name}_polar_contribution'] = contribution.tolist()
                entry[f'{name}_local_true_rival'] = z['local'][j, [true, rival]].tolist()
                entry[f'{name}_scores'] = z['frequency'][j].tolist()
                np.testing.assert_allclose(entry['base_margin'] + contribution.sum(), entry[f'{name}_margin'], atol=2e-6, rtol=1e-5)
            entry['reference_rows'] = refs.tolist()
            entry['reference_obs_ids'] = table.iloc[refs].obs_id.astype(int).tolist()
            entry['prediction_files'] = [str(SAVED/direction/f'seed{seed}'/arm/'source_predictions.npz') for arm in ('both', 'cosine_fixed')]
            fig, axes = plt.subplots(2, 5, figsize=(16, 7))
            provenance = []
            for col, idx in enumerate([row, *refs.tolist()]):
                for pol in range(2):
                    rec = records[idx]
                    p = display(axes[pol, col], rec, pol)
                    p.update(source_row=idx, original_obs_id=int(table.obs_id.iloc[idx]), sample_id=table.sample_id.iloc[idx],
                             role=table.role.iloc[idx], mmsi=table.mmsi.iloc[idx], pass_group=table.pass_group.iloc[idx])
                    provenance.append(p)
                    label = ('QUERY', 'TRUE1', 'TRUE2', 'RIVAL1', 'RIVAL2')[col]
                    axes[pol, col].set_title(f'{label} {("VV", "VH")[pol]} obs{int(table.obs_id.iloc[idx])}\n{CLASSES[int(table.class_id.iloc[idx])]}', fontsize=9)
                    axes[pol, col].set(xticks=[], yticks=[])
            fig.suptitle(f'{source.upper()} evaluation seed{seed}: both -> cosine_fixed, {outcome}\n'
                         f'{CLASSES[int(item.before_pred)]} -> {CLASSES[int(item.after_pred)]}; '
                         f'fixed-rival margin {item.before_margin:+.4f} -> {item.after_margin:+.4f}', fontsize=12)
            fig.text(.02, .02, 'Native 160px crop/pad, no input resizing; 8-bit model-input display chips, fixed gray 0..255. Green: supplied target OBB.', fontsize=9)
            fig.subplots_adjust(top=.82, bottom=.08, hspace=.22, wspace=.13)
            filename = f'{source}_seed{seed}_{outcome}_obs{obs}.png'
            fig.savefig(panels/filename, dpi=120)
            plt.close(fig)
            entry.update(image=f'panels/{filename}', provenance=provenance, helper=str(helper),
                         local_axes='[true/rival, VV/VH, body/side/end, similarity_sum/matched_mass]')
            cases.append(entry)
    write('cases.json', cases)
    write('panel_receipt.json', dict(panels=len(cases), absent_states=absent, actual_native_reference_images=True,
                                    visual_review_pending=True, comparison='both -> cosine_fixed frequency'))
    print([(x['source'], x['seed'], x['outcome'], x['obs_id'], x['image']) for x in cases], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('evaluate', 'cases'))
    args = parser.parse_args()
    (evaluate if args.mode == 'evaluate' else draw_cases)()
