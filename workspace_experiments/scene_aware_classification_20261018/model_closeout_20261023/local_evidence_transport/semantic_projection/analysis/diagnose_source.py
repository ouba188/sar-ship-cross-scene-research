"""Source utility_train cache diagnosis only; no fits or external inputs."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json
import textwrap
import numpy as np
import pandas as pd
import torch

OUT = Path(__file__).resolve().parent
P = OUT.parents[1]
UP = P.parent/'regional_utility_pilot/upstream'
sys.path.insert(0, str(P))
from match_readout import unit, operations
from extract_tokens import geo, slots


def body(x, mass):
    value = (x[:, :, :8]*mass[:, :, :8, None]).sum(2)
    return unit(value/np.maximum(mass[:, :, :8].sum(2)[..., None], 1e-8)).reshape(len(x), -1)


def main():
    torch.set_num_threads(2)
    table = pd.read_csv(UP/'table.csv', dtype={'mmsi': str, 'instance_id': str})
    take = np.flatnonzero(table.role.eq('utility_train'))
    y = table.iloc[take].class_id.to_numpy(int)
    z = np.load(P/'tokens.npz')
    np.testing.assert_array_equal(z['instance_ids'].astype(str), table.instance_id)
    f, mass, valid = z['layer1'], z['mass'], z['valid']
    pr = np.load(P/'projection_layer1.npz')
    sem = np.load(P/'semantic_projection/projection.npz')
    keys = {'PCA32': unit((f-pr['mean'])@pr['components'].T).astype(np.float32)}
    response = np.einsum('nptd,pcd->nptc', f, sem['W'])
    response -= response.mean(-1, keepdims=True)
    keys['class_contrast8'] = unit(response-sem['mean'][None, :, None]).astype(np.float32)
    for x in keys.values():
        x[mass <= 0] = 0
    caches = {'PCA32': np.load(P/'scores_layer1.npz'),
              'class_contrast8': np.load(P/'semantic_projection/scores_layer1.npz')}
    refs, bank = caches['PCA32']['refs'], pr['bank']
    np.testing.assert_array_equal(refs, caches['class_contrast8']['refs'])
    assert table.iloc[np.unique(bank)].role.eq('strategy_fit').all()
    for field in ('mmsi', 'pass_group'):
        assert not set(table.iloc[take][field]) & set(table.iloc[np.unique(bank)][field])
    scores = {k: v['independent'][:, :, :, 0, 0].mean(-1) for k, v in caches.items()}
    pa, pb = [scores[k][take].argmax(1) for k in keys]
    groups = {'pca_correct_contrast_wrong': (pa == y) & (pb != y),
              'both_wrong': (pa != y) & (pb != y)}
    cases = []
    for name, mask in groups.items():
        ids = take[mask]
        i = int(table.iloc[ids].sort_values('obs_id').index[0])
        true = int(table.iloc[i].class_id)
        r = scores['class_contrast8'][i].copy(); r[true] = -np.inf
        cases.append(dict(case=name, local_index=i, obs_id=int(table.iloc[i].obs_id),
                          true_class=true, rival_class=int(r.argmax()), group_n=len(ids), n=len(take)))
    pd.DataFrame(cases).to_csv(OUT/'case_selection.csv', index=False)
    rows = []
    for j, i in enumerate(take):
        for name in keys:
            s = scores[name][i]
            wrong = int(np.argmax(np.where(np.arange(8) == y[j], -np.inf, s)))
            av = valid[refs[i]].sum(1)
            rows.append(dict(local_index=int(i), obs_id=int(table.iloc[i].obs_id),
                class_id=int(y[j]), descriptor=name, prediction=int(s.argmax()), rival=wrong,
                true_margin=float(s[y[j]]-s[wrong]),
                true_valid_vv=int(av[y[j], 0]), true_valid_vh=int(av[y[j], 1]),
                rival_valid_vv=int(av[wrong, 0]), rival_valid_vh=int(av[wrong, 1]),
                all_four_query_ref_pol_valid=bool(valid[i].all() and valid[refs[i, y[j]]].all() and valid[refs[i, wrong]].all())))
    predictions = pd.DataFrame(rows)
    predictions.to_csv(OUT/'utility_direct_predictions.csv', index=False)
    bg = body(keys['class_contrast8'], mass)
    similarities = np.einsum('nd,ckd->nck', bg[take], bg[bank])
    picked = np.argsort(-similarities, axis=-1)[..., :2]
    semrefs = np.take_along_axis(np.broadcast_to(bank, (len(take), 8, 8)), picked, -1)
    overlap = (refs[take][..., :, None] == semrefs[..., None, :]).any(-1).sum(-1)
    rank_rows = []
    for j, i in enumerate(take):
        for c in range(8):
            order = bank[c][np.argsort(-similarities[j, c])]
            ranks = [int(np.flatnonzero(order == idx)[0])+1 for idx in refs[i, c]]
            rank_rows.append(dict(local_index=int(i), obs_id=int(table.iloc[i].obs_id),
                class_id=c, is_true=bool(c == y[j]), selected_overlap=int(overlap[j, c]),
                selected_semantic_rank1=ranks[0], selected_semantic_rank2=ranks[1]))
    pd.DataFrame(rank_rows).to_csv(OUT/'reference_key_rank.csv', index=False)
    key_metrics, case_pairs, margins, path_rows = [], [], [], []
    max_replay = 0.
    for case in cases:
        i, true, rival = case['local_index'], case['true_class'], case['rival_class']
        classes = np.array([true, rival]); r = bank[classes]
        for name, x in keys.items():
            with torch.inference_mode():
                v, t = operations(*[torch.tensor(a) for a in
                    (x[i:i+1], x[r][None], mass[i:i+1], mass[r][None])])
            for c, cls in enumerate(classes):
                chosen = np.array([int(np.flatnonzero(bank[cls] == idx)[0]) for idx in refs[i, cls]])
                saved = caches[name]['independent'][i, cls]
                replay = v['independent'][0, c, chosen].mean(0).numpy()
                max_replay = max(max_replay, float(abs(replay-saved).max()))
                np.testing.assert_allclose(replay, saved, atol=5e-5, rtol=1e-5)
                for k, idx in enumerate(r[c]):
                    for p, pol in enumerate(('vv', 'vh')):
                        sim = x[i, p, :8]@x[idx, p, :8].T
                        case_pairs.append(dict(case=case['case'], descriptor=name,
                            reference_class=int(cls), reference_index=int(idx), reference_obs=int(table.iloc[idx].obs_id),
                            pol=pol, selected=bool(idx in refs[i, cls]), valid=bool(valid[i, p] and valid[idx, p]),
                            mean_cosine=float(sim.mean()), max_cosine=float(sim.max()),
                            body_score=float(v['independent'][0, c, k, p, 0, 0]),
                            body_mass=float(v['independent'][0, c, k, p, 0, 1])))
            for p, pol in enumerate(('vv', 'vh')):
                local = x[i, p, :8]@x[i, p, :8].T
                key_metrics.append(dict(case=case['case'], descriptor=name, pol=pol,
                    within_query_body_cosine=float(local[np.triu_indices(8, 1)].mean()),
                    length=float(z['geometry'][i, p, 2]), width=float(z['geometry'][i, p, 3]),
                    body_neff_median=float(np.median(z['neff'][i, p, 1, :8]))))
            for arm in ('independent', 'anchored'):
                for region, ri in (('body', 0), ('side', 1), ('end', 2)):
                    value = caches[name][arm][i, classes, :, ri, 0]
                    margins.append(dict(case=case['case'], descriptor=name, arm=arm, region=region,
                        true_score=float(value[0].mean()), rival_score=float(value[1].mean()),
                        true_minus_rival=float((value[0]-value[1]).mean())))
    pd.DataFrame(case_pairs).to_csv(OUT/'case_all8_reference_scores.csv', index=False)
    pd.DataFrame(key_metrics).to_csv(OUT/'case_token_resolution.csv', index=False)
    pd.DataFrame(margins).to_csv(OUT/'case_region_margins.csv', index=False)
    os.environ['MPLCONFIGDIR'] = str(OUT/'.mplcache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    wanted = set(table.iloc[[c['local_index'] for c in cases]].obs_id)
    for c in cases:
        wanted.update(table.iloc[refs[c['local_index'], [c['true_class'], c['rival_class']]].ravel()].obs_id)
    snaps = {s['obs_id']: s for s in json.loads((geo.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8')) if s['obs_id'] in wanted}
    for case in cases:
        i, true, rival = case['local_index'], case['true_class'], case['rival_class']
        ids = [i, *refs[i, true], *refs[i, rival]]
        fig, axes = plt.subplots(2, 5, figsize=(23, 12))
        fig.subplots_adjust(left=.025, right=.985, bottom=.15, top=.84, wspace=.15, hspace=.36)
        for p, pol in enumerate(('vv', 'vh')):
            for j, idx in enumerate(ids):
                rr = table.iloc[idx]; pix, fp, g, place = geo.native(table, snaps, int(idx), p)
                ax = axes[p, j]; ax.imshow(pix, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
                ax.set_xlim(-.5,159.5);ax.set_ylim(159.5,-.5)
                if g is not None:
                    masks = slots(g,fp)
                    for rg,col in [(slice(0,8),'#24a878'),(slice(8,16),'#2c84c6'),(slice(16,20),'#d39222')]:
                        field=masks[rg].sum(0)
                        if field.min()<.5<field.max():ax.contour(field,levels=[.5],colors=[col],linewidths=.7)
                if not valid[idx,p]:ax.text(80,80,'NO LOCAL TOKENS',color='red',ha='center')
                label='QUERY' if j==0 else ('TRUE reference' if j<3 else 'WRONG reference')
                ax.set_title(f'{pol.upper()} {label}\nobs {int(rr.obs_id)} | '+textwrap.fill(rr.class_name,28)+'\n'+textwrap.fill(rr.port,28),fontsize=10)
                path_rows.append(dict(case=case['case'],column=j,pol=pol,local_index=int(idx),obs_id=int(rr.obs_id),instance_id=rr.instance_id,mmsi=rr.mmsi,class_id=int(rr.class_id),path=rr['image_'+pol],valid_local=bool(valid[idx,p])))
        msg=[]
        for name,s in scores.items():msg.append(f'{name}: pred {s[i].argmax()}, true-vs-shown-wrong {s[i,true]-s[i,rival]:+.6f}')
        fig.suptitle(f"SOURCE utility_train | {case['case']} | obs {case['obs_id']} | group {case['group_n']}/644\ntrue={true}, class-contrast strongest wrong={rival}; same actual PCA-selected references",fontsize=14)
        fig.text(.03,.045,'\n'.join(msg)+'\nDirect independent body similarity only: no fitted residual head, no B score.\nAll panels native160 crop/pad; RF35 mixes regions. Green body; blue side; amber end.',fontsize=12)
        fig.savefig(OUT/f"case_{case['obs_id']}.png",dpi=150);plt.close(fig)
    pd.DataFrame(path_rows).to_csv(OUT/'case_paths.csv',index=False)
    summary=dict(scope='utility_train only; source bank fit_train only; no external files',n=len(take),
        class_counts=table.iloc[take].class_id.value_counts().sort_index().to_dict(),
        pca_correct_contrast_wrong=int(((pa==y)&(pb!=y)).sum()),contrast_correct_pca_wrong=int(((pa!=y)&(pb==y)).sum()),
        both_wrong=int(((pa!=y)&(pb!=y)).sum()),both_correct=int(((pa==y)&(pb==y)).sum()),
        mean_top2_overlap=float(overlap.mean()),identical_top2_fraction=float((overlap==2).mean()),
        case_replay_max_abs=max_replay,new_fits=0,new_backbone_forwards=0)
    (OUT/'diagnostic_receipt.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
    print(pd.DataFrame(cases).to_string(index=False),flush=True)


if __name__=='__main__':main()
