"""Source-held benefit gate for same-vessel teacher case geometry."""

from full_teacher_data import BASE, ROOT, OUT as TEACHERS, PORTS, write, fit_head
import argparse
import gc
import time
import numpy as np
import pandas as pd
from unified_head_expert import ExpertContext, _conditions
from unified_head_core import score
from full_teacher_ops import numpy_phi
from check_translation_teacher_feasibility import make_cells, motion_features, pairs_for, margins
from check_teacher_weight_strength import local_weights
from incidence_transport import pair_weights

OUT = BASE/'case_relation_headroom_20261012'
SEEDS = (31415, 27182)
ARMS = ('single', 'equal_r18', 'knowledge_r18', 'equal_full', 'knowledge_full')


def load_motion(meta):
    ids = meta.obs_id.to_numpy()
    motion = pd.read_csv(BASE/'motion_knowledge_20261011/motion_knowledge.csv').set_index('obs_id').loc[ids]
    grid = pd.read_csv(BASE/'interaction_feasibility_20261012/geometry_join.csv').set_index('obs_id').loc[ids]
    np.testing.assert_array_equal(motion.mmsi, meta.mmsi)
    np.testing.assert_array_equal(motion['product'], meta['product'])
    sog = motion.sog_knots.to_numpy(float)
    cog = motion.cog_deg.to_numpy(float).copy()
    cog[~motion.cog_deg_valid.to_numpy(bool)] = np.nan
    az = grid.range_azimuth.to_numpy(float).copy()
    az[grid.grid_status.to_numpy() != 'ok'] = np.nan
    valid = motion.sog_knots_valid.to_numpy(bool) & motion.match_status.eq('matched').to_numpy()
    valid &= np.isfinite(sog) & (sog >= 0) & (sog < 102.2)
    return motion_features(sog, cog, az, valid)


def aggregate(meta, phi, motion, beta):
    """Other products only; product-balanced donors and unchanged unknown mass."""
    cells, x, means, members = make_cells(meta, phi, motion, np.zeros(len(meta), int))
    quality = x[:, 1:5]@beta
    equal, knowledge = phi.copy(), phi.copy()
    multi, active = np.zeros(len(meta), bool), np.zeros(len(meta), bool)
    donor_counts = np.zeros(len(meta), int)
    for _, group in cells.groupby('mmsi', sort=True):
        ix = group.index.to_numpy()
        for focal in ix:
            donor = ix[ix != focal]
            if not len(donor):
                continue
            rows = members[focal]
            w = np.ones(len(donor))
            known = cells.loc[donor, 'known'].to_numpy(bool)
            if known.sum() >= 2:
                local = local_weights(quality[donor[known]], 'minmax')
                w[known] = local/local.mean()
                active[rows] = True
            equal[rows] = means[donor].mean(0)
            knowledge[rows] = w@means[donor]/len(donor)
            multi[rows] = True
            donor_counts[rows] = len(donor)
    return equal, knowledge, multi, active, donor_counts


def r18_columns():
    # Retained SSL64 and original-coordinate SSL complement512; fixed R50 unaffected.
    return np.r_[np.arange(128, 192), np.arange(192+2048, 192+2560)]


def match_energy(raw, teacher, support, columns):
    """Keep the raw support centroid and centered energy; no query statistics."""
    result = teacher.copy()
    rc = raw[support][:, columns].mean(0)
    tc = teacher[support][:, columns].mean(0)
    re = np.mean(np.sum((raw[support][:, columns]-rc)**2, axis=1))
    te = np.mean(np.sum((teacher[support][:, columns]-tc)**2, axis=1))
    gain = np.sqrt(re/te) if te > 1e-20 else 1.
    result[:, columns] = (teacher[:, columns]-tc)*gain+rc
    return result, dict(raw_energy=float(re), teacher_energy=float(te), scale=float(gain))


def metric(y, pred):
    return dict(oa=float(100*np.mean(y == pred)),
                ba=float(100*np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)])))


def self_test():
    meta = pd.DataFrame(dict(mmsi=[1,1,1,1,1,2], product=['a','a','b','c','d','e'],
        class_id=[0]*6, port=['p']*6, incidence_deg=[30.]*6))
    phi = np.c_[[100.,100.,2.,8.,11.,77.], np.ones(6)]
    motion = np.zeros((6,6)); motion[3,0] = 1.; motion[4] = np.nan
    eq, kn, multi, active, _ = aggregate(meta, phi, motion, np.array([1.,0,0,0]))
    np.testing.assert_allclose(eq[:2,0], 7.)
    np.testing.assert_allclose(kn[:2,0], ((2/3)*2+(4/3)*8+11)/3)
    np.testing.assert_array_equal(eq[5], phi[5])
    assert not multi[5] and active[:2].all()
    poisoned = phi.copy(); poisoned[:2] = -999
    ep, kp, *_ = aggregate(meta, poisoned, motion, np.array([1.,0,0,0]))
    np.testing.assert_array_equal(ep[:2], eq[:2]); np.testing.assert_array_equal(kp[:2], kn[:2])
    changed, _ = match_energy(phi, eq, np.array([0,2,3,4,5]), np.array([0]))
    support = np.array([0,2,3,4,5])
    np.testing.assert_allclose(changed[support,0].mean(), phi[support,0].mean())
    np.testing.assert_allclose(changed[support,0].var(), phi[support,0].var())
    np.testing.assert_array_equal(changed[:,1], phi[:,1])
    print('TESTS PASS: product exclusion, unknown mass, fallback, support energy', flush=True)


def prepare(port, meta, cohorts, incidence):
    dest = OUT/port; dest.mkdir(parents=True, exist_ok=True)
    context = ExpertContext(ROOT/'learned_frequency_residual_20261009/tasks/self24ports'/port)
    context._prepare_source()
    ids = np.r_[cohorts['fit_obs_ids'], cohorts['held_obs_ids']]
    mm = meta.iloc[ids].reset_index(drop=True)
    pos = pd.Index(context.source_rows).get_indexer(mm.manifest_row)
    assert (pos >= 0).all()
    from full_teacher_data import SOURCE
    manifest = pd.read_csv(SOURCE/'manifest.csv', usecols=['class_id','mmsi','port'])
    held_ships = meta.iloc[cohorts['held_obs_ids']].mmsi.unique()
    train = ~np.isin(manifest.iloc[context.source_rows].mmsi, held_ships)
    assert not set(manifest.iloc[context.source_rows[train]].mmsi) & set(held_ships)
    head = fit_head(context.augmented[train], context.source_labels[train], context.transforms)
    prior = context.augmented[pos]@head['coef']+head['intercept']
    phi = numpy_phi(np.asarray(context.bank[pos]), context.raw[mm.manifest_row.to_numpy()],
                    context.extra[pos], incidence[mm.manifest_row.to_numpy()], context.geometry)
    nfit = len(cohorts['fit_obs_ids'])
    motion = load_motion(mm)
    with np.load(BASE/'crossfit_evidence_20261011/source_folds.npz') as f:
        np.testing.assert_array_equal(f['fit_obs_ids'], cohorts['fit_obs_ids'])
        folds = f['assignment']
    fit_scores = np.zeros((nfit,8))
    for fold in range(3):
        val = folds == fold
        excluded = mm.iloc[:nfit].loc[val,'mmsi'].unique()
        use = train & ~np.isin(manifest.iloc[context.source_rows].mmsi, excluded)
        reader = fit_head(context.augmented[use], context.source_labels[use], context.transforms)
        fit_scores[val] = context.augmented[pos[:nfit][val]]@reader['coef']+reader['intercept']
    cells, x, cs, _ = make_cells(mm.iloc[:nfit], fit_scores, motion[:nfit], folds)
    a,b = pairs_for(cells).T
    dx = x[b,1:5]-x[a,1:5]
    margin = margins(cs,cells.class_id.to_numpy())
    weights = pair_weights(cells,np.c_[a,b])
    beta = np.linalg.solve(dx.T@(weights[:,None]*dx)+.1*np.eye(4),
                           dx.T@(weights*(margin[b]-margin[a])))
    maps = {'single': phi}
    mask, active, counts = np.zeros(len(mm), bool), np.zeros(len(mm), bool), np.zeros(len(mm), int)
    equal, knowledge = phi.copy(), phi.copy()
    for role_ix in (np.arange(nfit), np.arange(nfit, len(mm))):
        a,b,m,k,n = aggregate(mm.iloc[role_ix].reset_index(drop=True), phi[role_ix], motion[role_ix], beta)
        equal[role_ix], knowledge[role_ix] = a,b
        mask[role_ix], active[role_ix], counts[role_ix] = m,k,n
    for prefix, value in [('equal',equal), ('knowledge',knowledge)]:
        maps[prefix+'_full'] = value
        narrow = phi.copy(); narrow[:, r18_columns()] = value[:, r18_columns()]
        maps[prefix+'_r18'] = narrow
    assert np.allclose(phi[:,-1],1.)
    write(dest/'COHORT.json', dict(fit_rows=nfit, held_rows=len(mm)-nfit,
        prior_fit_rows=int(train.sum()), excluded_prior_ships=len(held_ships),
        fit_multi=int(mask[:nfit].sum()), held_multi=int(mask[nfit:].sum()),
        held_reweightable=int(active[nfit:].sum()), phi_dimension=phi.shape[1],
        trainable_r18_coordinates=len(r18_columns()), geometry_context=port,
        held_ports=mm.iloc[nfit:].port.value_counts().to_dict(),
        knowledge_beta='Same4 fixed inputs/ridge0.1; refit from source-fit vessel-OOF scores, all heads exclude held ships',
        historical_encoder_and_geometry_saw_source=True))
    np.savez_compressed(dest/'maps.npz', obs_ids=ids, prior=prior, **maps,
        multi=mask, active=active, donor_counts=counts, beta=beta)
    del context; gc.collect()
    return ids, prior, maps, mask, active, counts


def run(port, meta, cohorts, incidence):
    ids, prior, maps, multi, active, donor_counts = prepare(port, meta, cohorts, incidence)
    dest = OUT/port; index = pd.Index(ids)
    y = meta.iloc[ids].class_id.to_numpy()
    results, changes, classes, diagnostics = [], [], [], []
    for seed in SEEDS:
        for draw in range(3):
            with np.load(BASE/'incidence_transport_20261011'/f'{seed}_source_{draw}_predictions.npz') as old:
                so, qo = old['support_obs_ids'], old['query_obs_ids']
            s, q = index.get_indexer(so), index.get_indexer(qo)
            assert (s >= 0).all() and (q >= 0).all() and len(s) == 64
            assert set(np.r_[so,qo]).issubset(set(cohorts['held_obs_ids']))
            assert not set(meta.iloc[so].mmsi) & set(meta.iloc[qo].mmsi)
            baseline = score(maps['single']@maps['single'][s].T, prior, s, y[s])[0][q]
            bp = baseline.argmax(1)
            for arm in ARMS:
                cols = r18_columns() if arm.endswith('r18') else np.arange(maps['single'].shape[1]-1)
                for scaling in ('native','matched'):
                    if arm == 'single' and scaling == 'matched':
                        continue
                    phi = maps[arm]
                    if scaling == 'matched':
                        phi, energy = match_energy(maps['single'], phi, s, cols)
                    else:
                        energy = dict(raw_energy=np.nan, teacher_energy=np.nan, scale=1.)
                    k = phi@phi[s].T
                    scores, alpha = score(k, prior, s, y[s])
                    pred = scores[q].argmax(1)
                    np.savez_compressed(dest/f'{seed}_{draw}_{arm}_{scaling}.npz',
                        support_obs_ids=so, query_obs_ids=qo, support_labels=y[s],
                        scores=scores[q], baseline=baseline, kernel_support=k[s],
                        kernel_query=k[q], prior_support=prior[s], prior_query=prior[q])
                    for cohort, m in [('all',np.ones(len(q),bool)),('multi',multi[q]),
                                      ('single',~multi[q]),('knowledge_active',active[q])]:
                        if not m.any():
                            continue
                        results.append(dict(context=port,seed=seed,draw=draw,arm=arm,scaling=scaling,
                            cohort=cohort,n=int(m.sum()),**metric(y[q][m],pred[m]),
                            rescue=int(np.sum((bp[m]!=y[q][m])&(pred[m]==y[q][m]))),
                            harm=int(np.sum((bp[m]==y[q][m])&(pred[m]!=y[q][m])))))
                    for c in np.unique(y[q]):
                        m = y[q] == c
                        classes.append(dict(context=port,seed=seed,draw=draw,arm=arm,scaling=scaling,
                            class_id=int(c),n=int(m.sum()),recall=float(100*np.mean(pred[m]==c))))
                    for j in np.flatnonzero((bp == y[q]) != (pred == y[q])):
                        row = meta.iloc[qo[j]]
                        changes.append(dict(context=port,seed=seed,draw=draw,arm=arm,scaling=scaling,
                            obs_id=int(qo[j]),mmsi=str(row.mmsi),port=row.port,class_id=int(y[q[j]]),
                            before=int(bp[j]),after=int(pred[j]),rescue=bool(pred[j] == y[q[j]]),
                            donors=int(donor_counts[q[j]])))
                    nearest = np.argmax(k[q]/np.maximum(np.linalg.norm(phi[s],axis=1),1e-12),axis=1)
                    diagnostics.append(dict(context=port,seed=seed,draw=draw,arm=arm,scaling=scaling,
                        nearest_support_accuracy=float(100*np.mean(y[s][nearest]==y[q])),
                        score_shift_rms=float(np.sqrt(np.mean((scores[q]-baseline)**2))),
                        support_multi=int(multi[s].sum()),query_multi=int(multi[q].sum()),**energy))
            print('EVALUATED', port, seed, draw, flush=True)
    for name, rows in [('metrics',results),('events',changes),('classes',classes),('diagnostics',diagnostics)]:
        pd.DataFrame(rows).to_csv(dest/f'{name}.csv',index=False)


def report():
    frames = {name:pd.concat([pd.read_csv(OUT/p/f'{name}.csv') for p in PORTS],ignore_index=True)
              for name in ('metrics','events','classes','diagnostics')}
    for name, frame in frames.items():
        frame.to_csv(OUT/f'{name}.csv',index=False)
    m = frames['metrics']
    summary = m.groupby(['cohort','arm','scaling'])[['oa','ba','rescue','harm']].mean()
    summary.to_csv(OUT/'summary.csv')
    m.groupby(['context','arm','scaling','cohort'])[['oa','ba']].mean().to_csv(OUT/'context_summary.csv')
    write(OUT/'COMPLETE.json',dict(contexts=list(PORTS),support_draws=6,neural_fits=0,
        target_evaluations=0,geometry_contexts_not_independent_datasets=True,
        full_expert_solver=True,final_fusion_not_evaluated=True,source_held_reused=True))
    print(summary.loc['all'].to_string(),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-only',action='store_true')
    parser.add_argument('--report-only',action='store_true')
    parser.add_argument('--ports',nargs='+',default=list(PORTS))
    args = parser.parse_args()
    self_test()
    if args.test_only:
        return
    if args.report_only:
        report(); return
    OUT.mkdir(exist_ok=True)
    write(OUT/'PROTOCOL.json',dict(date='2026-10-12',phase='Headroom before student module',
        arms=list(ARMS),primary='equal_r18, matched support energy, entire held cohort OA/BA',
        representation='Exact deployed retained/complement/stable/drift/incidence expert phi',
        prior='New matched source head excludes all 302 held ships; shared across all interventions',
        supports='Original source-held 64 support cases x 2 seeds x 3 draws; queries exclude support ships',
        donors='Same ship other whole products within fit or held role separately, product-balanced',
        knowledge='Same4 fixed inputs/ridge0.1, refit source-fit OOF quality reader; heads exclude all held ships',
        scales='Native diagnostic and support-only centroid/centered-energy matching; ridge fixed8',
        fallback='No other product uses unchanged observation; report all samples and coverage',
        scope='Privileged teacher feasibility only, NOT deployable inference, NOT full fusion results',
        limitation='Historical encoder/PCA/geometry exposed to source; linear head vessel-held only; repeated development cohorts',
        no_target_runs=True,no_backbone_training=True))
    meta = pd.read_csv(BASE/'observations.csv')
    with np.load(BASE/'incidence_transport_20261011/source_cohorts.npz') as z:
        cohorts = {k:z[k] for k in ('fit_obs_ids','held_obs_ids')}
    assert not set(meta.iloc[cohorts['fit_obs_ids']].mmsi) & set(meta.iloc[cohorts['held_obs_ids']].mmsi)
    incidence = _conditions()
    for port in args.ports:
        assert port in PORTS
        run(port, meta, cohorts, incidence)
    if all((OUT/p/'metrics.csv').exists() for p in PORTS):
        report()


if __name__ == '__main__':
    main()
