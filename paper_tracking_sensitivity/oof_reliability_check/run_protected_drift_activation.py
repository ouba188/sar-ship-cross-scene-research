"""Matched 2x2 trial of protected drift attenuation and explicit stable readout."""

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '2'

import argparse
import gc
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from evidence_aware_handoff import fit_gate, predict_gate, route_scores
from projection_residual_expert import discarded_coordinates, fit_kernel_correction
from protected_drift_activation import protected_drift_basis, fit_source_heads
from receiver_consistent_handoff import confidence_parts
from residual_expert import fit_correction
from run_generic_transfer import metrics
from sar_joint_completion import SOURCE, write_json


ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
OUT = ROOT/'protected_drift_activation_20261008'
PARENT = ROOT/'module_completion_20261008/ports_projection'
CANDIDATE = ROOT/'conditional_invariance_probe_20261008'
TRACE = ROOT/'latent_usage_attribution_20261008'
EVIDENCE = ROOT/'evidence_aware_handoff_20261008'
BASE = ROOT/'oof_base_adapt_results_20261007'
FACTORS = dict(matched_baseline=(False, False), drift_only=(True, False),
               stable_only=(False, True), joint=(True, True))
ARMS = ('original', *FACTORS)
DRIFT_STRENGTH = .5
DRIFT_RANK = 16
STABLE_KERNEL = .25


def archive(path):
    with np.load(path, allow_pickle=False) as saved:
        return {k: saved[k] for k in saved.files}


def protocol():
    record = dict(date='2026-10-08', task='Protected drift suppression and explicit stable activation factorial trial',
        arms=list(ARMS), labels_per_target_draw=64, draws=5, budget=.05,
        stable='Frozen source-discovered R50 candidate span Qs, not a proven invariant representation',
        drift='Equal-class/equal-port centered class means; remove Qs overlap; top16 residual singular directions',
        operation='x_new=x-.5*(x Qd)Qd.T in source-standardized raw R50; Qs.T Qd=0',
        strength=DRIFT_STRENGTH, rank=DRIFT_RANK,
        activation='Append source-standardized Qs coordinates to source base/expert classification readouts',
        source_branch_scale='sqrt(128/r) for base; sqrt(192/r) for expert; core unchanged',
        base_adaptation='Original residual features plus stable/sqrt(r) when active; one shared bias; ridge8',
        expert_adaptation='Original .5 retained+.5 complement+1 kernel plus .25 stable kernel when active; ridge8',
        source_heads='Class-balanced ridge1 refitted in all factorial arms on current frozen R18 features',
        gate='Four original margin/entropy inputs; source trained separately for each receiving pair; LR C1',
        source_gate_training='22 source episodes per outer; original identity/product-purged head train rows and64supports',
        geometry_boundary='Geometry/Qs/Qd/scales and M3 encoder fit on ALL outer source; only head readouts inner-held',
        not_claimed='NOT fully inner-OOF geometry or encoder; outer target labels only original64 adaptation',
        normalization='Common pre-attenuation source scales; do not renormalize attenuation away',
        source_fixed_parameters=True, target_query_selection=False,
        original='Historical main model preserved separately; matched baseline includes head/gate refit effects',
        freezes='All23 target policies before final query readout',
        compute='Frozen R50 and R18 backbones, CPU readout experiment; not end-to-end learned invariance',
        independent_confirmation=False, main_model_unchanged=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT/'PROTOCOL.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != record:
            raise ValueError('Changed protocol requires a new output directory')
    else:
        write_json(path, record)


def cell_means(raw, rows, labels, domains, center, scale):
    means, classes = [], []
    for c in np.unique(labels[rows]):
        groups = []
        for port in np.unique(domains[rows]):
            chosen = rows[(labels[rows] == c) & (domains[rows] == port)]
            if len(chosen) >= 5:
                total = np.zeros(raw.shape[1], dtype=float)
                for lo in range(0, len(chosen), 512):
                    total += raw[chosen[lo:lo+512]].sum(0, dtype=float)
                groups.append((total/len(chosen)-center)/scale)
        if len(groups) >= 3:
            means.extend(groups)
            classes.extend([c]*len(groups))
    return np.asarray(means), np.asarray(classes)


def project_latents(raw, rows, center, scale, basis):
    values = np.empty((len(rows), basis.shape[1]))
    for lo in range(0, len(rows), 512):
        x = (np.asarray(raw[rows[lo:lo+512]], dtype=float)-center)/scale
        values[lo:lo+len(x)] = x@basis
    return values


def make_transforms(g):
    r, d = g['stable_rank'], g['drift_rank']
    n = 192+r+d
    answer = {}
    for arm, (suppress, activate) in FACTORS.items():
        strength = DRIFT_STRENGTH if suppress else 0.
        for head, width in (('base', 128), ('expert', 192)):
            transform = np.zeros((n, width+(r if activate else 0)))
            if head == 'base':
                transform[:128,:128] = np.eye(128)
                transform[192+r:,:128] = -strength*g['drift_to_xs']
            else:
                transform[:192,:192] = np.diag(1./g['model']['expert_scale'])
                transform[192+r:,:128] = -strength*g['drift_to_xs']/g['model']['expert_scale'][:128]
            if activate:
                transform[192:192+r,width:] = np.eye(r)*np.sqrt(width/r)
            answer[arm+'_'+head] = transform
    return answer


def prepare_port(port, raw, frame):
    folder = OUT/'ports'/port
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'GEOMETRY_READY.json').exists():
        with (folder/'geometry.pkl').open('rb') as f:
            return pickle.load(f)
    old = archive(PARENT/port/'matched_joint/joint_policies.npz')
    candidate = archive(CANDIDATE/'ports'/port/'frozen.npz')
    with np.load(TRACE/'ports'/port/'trace.npz') as saved:
        qs = saved['q']
    source, query = old['source_rows'], old['query_rows']
    np.testing.assert_array_equal(source, candidate['source_rows'])
    np.testing.assert_array_equal(query, candidate['query_rows'])
    labels, domains = frame.class_id.to_numpy(), frame.port.to_numpy()
    if np.any(domains[source] == port) or np.intersect1d(source, query).size:
        raise ValueError('Outer target enters geometry fitting')
    center, scale = candidate['feature_center'], candidate['feature_scale']
    means, classes = cell_means(raw, source, labels, domains, center, scale)
    drift = protected_drift_basis(means, classes, qs, rank=DRIFT_RANK)
    qd = drift['q_drift']
    with (PARENT/port/'source_model.pkl').open('rb') as f:
        model = pickle.load(f)
    stats = archive(PARENT/port/'deployment_complement.npz')
    adapted = archive(BASE/port/'policies.npz')
    basis = np.column_stack((qs, qd))
    latent_source = project_latents(raw, source, center, scale, basis)
    latent_target = project_latents(raw, query, center, scale, basis)
    r, d = qs.shape[1], qd.shape[1]
    zcenter = latent_source[:,:r].mean(0)
    zscale = latent_source[:,:r].std(0)+1e-6
    maps = (qd.T*scale[None,:])
    dxs = (maps@model['pca'].components_.T)/model['scale']
    dcomp = (maps-(maps@model['pca'].components_.T)@model['pca'].components_)
    dcomp /= stats['discarded_scale'][:2048]
    np.testing.assert_allclose(qs.T@qd, 0, atol=1e-8)
    source_extra_dir = EVIDENCE/'source_features'/port
    extra_rows = np.load(source_extra_dir/'rows.npy')
    np.testing.assert_array_equal(source, extra_rows)
    extra_source = np.load(source_extra_dir/'features.npy', mmap_mode='r')
    extra_target = np.load(PARENT/port/'matched_joint/joint_extra.npy', mmap_mode='r')
    with np.load(ROOT/'pred_cache2'/f'{port}.npz') as saved:
        xs = saved['Xs']
    for name, ids, extra, latent in (('source', source, extra_source, latent_source),
                                    ('target', query, extra_target, latent_target)):
        ssl = model['pca_ssl'].transform(extra).astype(np.float32)
        bank = np.column_stack((xs[ids], ssl, (latent[:,:r]-zcenter)/zscale, latent[:,r:]))
        np.save(folder/(name+'_bank.npy'), bank)
        np.save(folder/(name+'_rows.npy'), ids)
    g = dict(model=model, stats=stats, qs=qs, qd=qd, stable_rank=r, drift_rank=d,
        feature_center=center, feature_scale=scale, stable_center=zcenter, stable_scale=zscale,
        drift_to_xs=dxs, drift_to_complement=dcomp,
        base_center=adapted['source_center'], base_scale=adapted['source_scale'])
    with (folder/'geometry.pkl').open('wb') as f:
        pickle.dump(g, f)
    write_json(folder/'GEOMETRY_READY.json', dict(port=port, source_rows=len(source), query_rows=len(query),
        stable_rank=r, drift_rank=d, overlap=float(np.max(np.abs(qs.T@qd))) if d else 0.,
        selected_energy_fraction=float(drift['selected_energy_fraction']),
        nuisance_energy_in_stable_fraction=float(drift['nuisance_energy_in_stable_fraction']),
        selected_drift_amplitude_ratio=.5, selected_drift_energy_ratio=.25,
        target_labels_fitted=0, source_geometry_inner_held=False))
    print(f'GEOMETRY {port}: stable={r}, drift={d}', flush=True)
    return g


def make_views(bank, raw_values, extra, g):
    model, stats = g['model'], g['stats']
    r = g['stable_rank']
    stable = np.asarray(bank[:,192:192+r])/np.sqrt(r)
    zd = np.asarray(bank[:,192+r:])
    comp = np.column_stack((discarded_coordinates(raw_values, model['pca']),
                            discarded_coordinates(extra, model['pca_ssl'])))
    comp = (comp-stats['discarded_center'])/stats['discarded_scale']
    common, phis = {}, {}
    for suppress in (False, True):
        if suppress:
            core = np.asarray(bank[:,:128])-DRIFT_STRENGTH*(zd@g['drift_to_xs'])
            complementary = comp.copy()
            complementary[:,:2048] -= DRIFT_STRENGTH*(zd@g['drift_to_complement'])
        else:
            core = np.asarray(bank[:,:128], dtype=np.float32)
            complementary = comp
        xe = (np.column_stack((core, np.asarray(bank[:,128:192], dtype=np.float32)))
              -model['expert_center'])/model['expert_scale']
        retained = (xe/np.sqrt(192)).astype(float)
        common[suppress] = dict(retained=retained, complement=complementary/np.sqrt(2560))
        base = (core-g['base_center'])/g['base_scale']/np.sqrt(128)
        for activate in (False, True):
            body = np.column_stack((base, stable)) if activate else base
            phis[suppress,activate] = np.column_stack((body, np.ones(len(bank))))
    return dict(common=common, phis=phis, stable=stable)


def predict_draw(bank, views, heads, support, y):
    kernels = {}
    for suppress, value in views['common'].items():
        retained, comp = value['retained'], value['complement']
        kernels[suppress] = .5*(retained@retained[support].T)+.5*(comp@comp[support].T)+1.
    stable_kernel = STABLE_KERNEL*(views['stable']@views['stable'][support].T)
    records = {}
    for arm, factors in FACTORS.items():
        suppress, activate = factors
        base_head, expert_head = heads[arm+'_base'], heads[arm+'_expert']
        bprior = bank@base_head['coef']+base_head['intercept']
        eprior = bank@expert_head['coef']+expert_head['intercept']
        phi = views['phis'][factors]
        bcoef = fit_correction(phi[support], y, bprior[support])
        base = bprior+phi@bcoef
        kernel = kernels[suppress]+(stable_kernel if activate else 0.)
        ecoef = fit_kernel_correction(kernel[support], y, eprior[support])
        records[arm] = dict(base=base, expert=eprior+kernel@ecoef,
                            base_coefficients=bcoef, expert_coefficients=ecoef)
    return records


def gate_features(base, expert):
    bm, bh = confidence_parts(base)
    em, eh = confidence_parts(expert)
    return np.column_stack((em, bm, eh, bh))


def train_source(port, g, raw, labels):
    folder = OUT/'ports'/port
    if (folder/'SOURCE_FITTED.json').exists():
        with (folder/'source_models.pkl').open('rb') as f:
            return pickle.load(f)
    started = time.monotonic()
    source_rows = np.load(folder/'source_rows.npy')
    bank = np.load(folder/'source_bank.npy', mmap_mode='r')
    extra = np.load(EVIDENCE/'source_features'/port/'features.npy', mmap_mode='r')
    transforms = make_transforms(g)
    episodes = json.loads((PARENT/port/'EPISODES_COMPLETE.json').read_text(encoding='utf-8'))['episodes']
    groups = {arm: {key: [] for key in ('base', 'expert', 'labels', 'eligible', 'weights', 'features')}
              for arm in FACTORS}
    source_metrics = []
    for i, item in enumerate(episodes):
        with np.load(item['path']) as saved:
            train, rows, support, query = saved['train_rows'], saved['rows'], saved['support'], saved['query']
        train_local, held_local = np.searchsorted(source_rows, train), np.searchsorted(source_rows, rows)
        np.testing.assert_array_equal(source_rows[train_local], train)
        np.testing.assert_array_equal(source_rows[held_local], rows)
        if np.intersect1d(train, rows).size or np.intersect1d(support, query).size:
            raise ValueError('Source episode head/support/query overlap')
        heads = fit_source_heads(bank[train_local], labels[train], transforms)
        held_bank = np.asarray(bank[held_local])
        views = make_views(held_bank, raw[rows], extra[held_local], g)
        result = predict_draw(held_bank, views, heads, support, labels[rows[support]])
        y = labels[rows]
        _, inverse, counts = np.unique(y[query], return_inverse=True, return_counts=True)
        weights = 1./(len(episodes)*len(counts)*counts[inverse])
        for arm, pred in result.items():
            b, e = pred['base'], pred['expert']
            observed = np.bincount(y[support], minlength=8) > 0
            eligible = (b.argmax(1) != e.argmax(1)) & observed[e.argmax(1)]
            values = dict(base=b[query], expert=e[query], labels=y[query], eligible=eligible[query],
                          weights=weights, features=gate_features(b,e)[query])
            for key, value in values.items():
                groups[arm][key].append(value)
            bm, em = metrics(y[query], b[query].argmax(1), 8), metrics(y[query], e[query].argmax(1), 8)
            source_metrics.append(dict(port=port, held_source=item['port'], arm=arm,
                base_ba=bm['ba'], base_acc=bm['acc'], expert_ba=em['ba'], expert_acc=em['acc']))
        del heads, views, held_bank, result
        if (i+1) % 6 == 0:
            print(f'SOURCE {port}: {i+1}/{len(episodes)} episodes, {time.monotonic()-started:.1f}s', flush=True)
    gates = {}
    for arm in FACTORS:
        data = {k: np.concatenate(v) for k,v in groups[arm].items()}
        gates[arm] = fit_gate(data['features'], data['base'], data['expert'], data['labels'], data['eligible'], data['weights'])
    heads = fit_source_heads(np.asarray(bank), labels[source_rows], transforms)
    fitted = dict(heads=heads, gates=gates)
    with (folder/'source_models.pkl').open('wb') as f:
        pickle.dump(fitted, f)
    pd.DataFrame(source_metrics).to_csv(folder/'source_episode_metrics.csv', index=False, encoding='utf-8-sig')
    write_json(folder/'SOURCE_FITTED.json', dict(port=port, episodes=len(episodes),
        arms=list(FACTORS), gate_features=4, geometry_inner_held=False,
        target_labels_used=0, elapsed_seconds=time.monotonic()-started))
    print(f'SOURCE READY {port}: {time.monotonic()-started:.1f}s', flush=True)
    return fitted


def freeze_target(port, g, fitted, raw, labels):
    folder = OUT/'ports'/port
    if (folder/'TARGET_FROZEN.json').exists():
        return
    old = archive(PARENT/port/'matched_joint/joint_policies.npz')
    bank = np.load(folder/'target_bank.npy')
    rows = np.load(folder/'target_rows.npy')
    np.testing.assert_array_equal(rows, old['query_rows'])
    extra = np.load(PARENT/port/'matched_joint/joint_extra.npy', mmap_mode='r')
    views = make_views(bank, raw[rows], extra, g)
    all_bases, all_experts, all_gates, all_routes = [], [], [], []
    correction_records = {}
    kernel_error = 0.
    for draw, ev in enumerate(old['eval_mask']):
        support = np.searchsorted(rows, old['support_rows'][draw])
        np.testing.assert_array_equal(rows[support], old['support_rows'][draw])
        if ev[support].any():
            raise ValueError('Target supports enter query')
        y = labels[old['support_rows'][draw]]
        preds = predict_draw(bank, views, fitted['heads'], support, y)
        # The no-transform kernel must reproduce the historical expert with its old prior.
        xe = (np.column_stack((bank[:,:128].astype(np.float32), bank[:,128:192].astype(np.float32)))
              -g['model']['expert_center'])/g['model']['expert_scale']
        prior = g['model']['expert'].decision_function(xe).astype(np.float32).astype(float)
        retained, comp = views['common'][False]['retained'], views['common'][False]['complement']
        kernel = .5*(retained@retained[support].T)+.5*(comp@comp[support].T)+1.
        replay = prior+kernel@fit_kernel_correction(kernel[support], y, prior[support])
        error = float(np.max(np.abs(replay-old['expert_scores'][draw])))
        kernel_error = max(kernel_error, error)
        np.testing.assert_allclose(replay, old['expert_scores'][draw], atol=2e-6, rtol=2e-6)
        bases = [old['base_scores'][draw]]
        experts = [old['expert_scores'][draw]]
        gates = [old['gate_scores'][draw]]
        routing = [dict(predictions=old['predictions'][draw], selected=old['selected'][draw],
            cap=old['cap'][draw], base_predictions=bases[0].argmax(1), expert_predictions=experts[0].argmax(1))]
        for arm, pred in preds.items():
            b, e = pred['base'], pred['expert']
            score = predict_gate(fitted['gates'][arm], gate_features(b,e))
            routed = route_scores(b,e,score,y,ev,rows)
            if routed['cap'] != old['cap'][draw] or routed['selected'][~ev].any():
                raise ValueError('Budget or target support exclusion changed')
            bases.append(b)
            experts.append(e)
            gates.append(score)
            routing.append(routed)
            for key in ('base_coefficients', 'expert_coefficients'):
                correction_records.setdefault(arm+'_'+key, []).append(pred[key])
        all_bases.append(np.stack(bases))
        all_experts.append(np.stack(experts))
        all_gates.append(np.stack(gates))
        all_routes.append(routing)
    np.savez_compressed(folder/'policies.npz', query_rows=rows, source_rows=old['source_rows'],
        support_rows=old['support_rows'], eval_mask=old['eval_mask'], arm_names=np.asarray(ARMS),
        base_scores=np.stack(all_bases), expert_scores=np.stack(all_experts), gate_scores=np.stack(all_gates),
        **{k: np.stack([[r[k] for r in draw] for draw in all_routes]) for k in all_routes[0][0]},
        **{k: np.stack(v) for k,v in correction_records.items()})
    write_json(folder/'TARGET_FROZEN.json', dict(port=port, arms=list(ARMS), draws=5,
        supports=64, budget=.05, original_preserved=True, unchanged_kernel_replay_error=kernel_error))
    print('TARGET FROZEN '+port, flush=True)


def evaluate(inventory, labels):
    rows, sources = [], []
    for item in inventory:
        port = item['port']
        folder = OUT/'ports'/port
        p = archive(folder/'policies.npz')
        y = labels[p['query_rows']]
        sources.append(pd.read_csv(folder/'source_episode_metrics.csv'))
        for draw, ev in enumerate(p['eval_mask']):
            for j, arm in enumerate(ARMS):
                pred, base, expert = p['predictions'][draw,j], p['base_predictions'][draw,j], p['expert_predictions'][draw,j]
                m, bm, em = metrics(y[ev], pred[ev], 8), metrics(y[ev], base[ev], 8), metrics(y[ev], expert[ev], 8)
                rows.append(dict(port=port, draw=draw, arm=arm, acc=m['acc'], ba=m['ba'],
                    base_acc=bm['acc'], base_ba=bm['ba'], expert_acc=em['acc'], expert_ba=em['ba'],
                    selected=int(p['selected'][draw,j].sum()), cap=int(p['cap'][draw,j]), query_rows=int(ev.sum()),
                    rescued=int(np.sum(ev & (pred == y) & (base != y))),
                    harmed=int(np.sum(ev & (pred != y) & (base == y)))))
    df = pd.DataFrame(rows)
    df.to_csv(OUT/'metrics.csv', index=False, encoding='utf-8-sig')
    pd.concat(sources).to_csv(OUT/'source_episode_metrics.csv', index=False, encoding='utf-8-sig')
    pm = df.groupby(['port','arm']).mean(numeric_only=True).drop(columns='draw').reset_index()
    pm.to_csv(OUT/'port_means.csv', index=False, encoding='utf-8-sig')
    means = pm.groupby('arm').mean(numeric_only=True).reset_index()
    means.to_csv(OUT/'equal_port_means.csv', index=False, encoding='utf-8-sig')
    contrasts = []
    for a,b in (('matched_baseline','original'),('drift_only','matched_baseline'),
                ('stable_only','matched_baseline'),('joint','matched_baseline'),
                ('joint','drift_only'),('joint','stable_only'),('joint','original')):
        av, bv = pm[pm.arm == a].set_index('port'), pm[pm.arm == b].set_index('port')
        d = av.ba-bv.ba
        contrasts.append(dict(candidate=a, control=b, delta_ba_pp=float(100*d.mean()),
            delta_oa_pp=float(100*(av.acc-bv.acc).mean()),
            delta_base_ba_pp=float(100*(av.base_ba-bv.base_ba).mean()),
            delta_expert_ba_pp=float(100*(av.expert_ba-bv.expert_ba).mean()),
            positive_ports=int((d>1e-12).sum()), negative_ports=int((d < -1e-12).sum())))
    write_json(OUT/'COMPLETE.json', dict(ports=len(inventory), arms=list(ARMS), rows=len(df),
        means=means.to_dict('records'), contrasts=contrasts, main_model_unchanged=True,
        independent_confirmation=False, target_query_selection=False))
    print(means.to_string(index=False), flush=True)
    print(json.dumps(contrasts,indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ports', nargs='*')
    args = parser.parse_args()
    protocol()
    if (OUT/'COMPLETE.json').exists():
        raise ValueError('Preserve completed output')
    inventory = json.loads((BASE/'POLICIES_SAVED.json').read_text(encoding='utf-8'))
    frame = pd.read_csv(SOURCE/'manifest.csv', usecols=['class_id','port'], keep_default_na=False)
    labels = frame.class_id.to_numpy()
    raw = np.load(SOURCE/'pooled_features.npy', mmap_mode='r')
    with threadpool_limits(limits=2):
        for item in inventory:
            port = item['port']
            if args.ports and port not in args.ports:
                continue
            if (OUT/'ports'/port/'TARGET_FROZEN.json').exists():
                continue
            g = prepare_port(port, raw, frame)
            fitted = train_source(port, g, raw, labels)
            freeze_target(port, g, fitted, raw, labels)
            del g, fitted
            gc.collect()
    if all((OUT/'ports'/item['port']/'TARGET_FROZEN.json').exists() for item in inventory):
        write_json(OUT/'ALL_TARGETS_FROZEN.json', dict(ports=len(inventory), arms=list(ARMS)))
        evaluate(inventory, labels)
    else:
        print('PARTIAL: no target query metrics read', flush=True)


if __name__ == '__main__':
    main()
