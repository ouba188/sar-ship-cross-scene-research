"""Paired seed summaries, fixed-head decompositions, and native VV/VH panels."""
import json
import sys
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from run import ROOT, AXIS, LOCAL, ARMS, prior, old, Residual, geometry
SOURCE = ROOT


def main():
    f = pd.read_csv(prior.ROOT / 'cohort.csv', dtype={'mmsi': str})
    y = f.class_id.to_numpy()
    p = pd.read_csv(ROOT / 'predictions.csv', dtype={'mmsi': str})
    rows = []; paired = []
    for seed in range(3):
        for arm in ('baseline', *ARMS):
            scores = []; labels = []
            for fold in range(3):
                d = np.load(ROOT / f'scores_f{fold}_{ARMS[0] if arm == "baseline" else arm}_s{seed}.npz')
                ss = d['base'] if arm == 'baseline' else d['scores']
                scores.append(ss); labels.append(y[d['rows']])
            rows.append(dict(arm=arm, seed=seed, **prior.metrics(np.concatenate(labels), np.concatenate(scores))))
        for arm in ARMS:
            a = p[p.seed.eq(seed) & p.arm.eq(arm)]
            base_ok = a.base_prediction.eq(a.class_id)
            paired.append(dict(arm=arm, seed=seed, rescue=int((a.correct & ~base_ok).sum()), harm=int((~a.correct & base_ok).sum())))
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(ROOT / 'seed_summary.csv', index=False)
    per_seed.groupby('arm')[['oa', 'ba', 'brier']].agg(['mean', 'std']).to_csv(ROOT / 'summary.csv')
    pd.DataFrame(paired).to_csv(ROOT / 'paired.csv', index=False)

    comparisons = [('late_both', 'early_both'), ('late_both', 'keep_roles'),
                   ('keep_refs', 'early_both'), ('late_both', 'keep_refs'),
                   ('keep_roles', 'early_both'), ('late_both', 'late_image'), ('late_both', 'late_ship')]
    factorial = []
    for aa, bb in comparisons:
        for seed in range(3):
            a = per_seed.query('arm == @aa and seed == @seed').iloc[0]
            b = per_seed.query('arm == @bb and seed == @seed').iloc[0]
            factorial.append(dict(candidate=aa, reference=bb, seed=seed,
                delta_oa=a.oa-b.oa, delta_ba=a.ba-b.ba, delta_brier=a.brier-b.brier))
    pd.DataFrame(factorial).to_csv(ROOT / 'factorial_effects.csv', index=False)
    # Check whether the new samplers actually produce distinct descriptions.
    phys = np.load(SOURCE / 'late_both.npy', mmap_mode='r')
    img = np.load(SOURCE / 'late_image.npy', mmap_mode='r')
    ship = np.load(SOURCE / 'late_ship.npy', mmap_mode='r')
    descriptor_rows = []
    excluded = pd.read_csv(LOCAL / 'EXCLUSIONS.csv').obs_id.to_numpy()
    for fold in range(3):
        for partition, ii in zip(('train','val','test'), prior.indices(f,fold)):
            ii = ii[~f.iloc[ii].obs_id.isin(excluded).to_numpy()]
            pp = np.asarray(phys[ii]); denom = max(float(np.linalg.norm(pp)),1e-8)
            for name, other in (('image',img),('ship',ship)):
                delta = pp-np.asarray(other[ii])
                descriptor_rows.append(dict(fold=fold,partition=partition,reference=name,n=len(ii),
                    relative_l2=float(np.linalg.norm(delta))/denom,mean_abs=float(np.abs(delta).mean())))
    pd.DataFrame(descriptor_rows).to_csv(ROOT / 'descriptor_reference_effect.csv',index=False)
    for group in ('class_id', 'port'):
        g = p.groupby(['arm', 'seed', group]).agg(n=('correct', 'size'), oa=('correct', 'mean'))
        g.to_csv(ROOT / f'by_{group}.csv')
    coverage=pd.read_csv(SOURCE/'coverage.csv')
    available=coverage.pivot(index='obs_id',columns='pol',values='radar_valid').fillna(0).rename(columns={'vv':'radar_vv','vh':'radar_vh'})
    pp=p.join(available,on='obs_id'); pp['base_correct']=pp.base_prediction.eq(pp.class_id)
    pp['rescue']=pp.correct & ~pp.base_correct; pp['harm']=~pp.correct & pp.base_correct
    pp.groupby(['arm','seed','radar_vv','radar_vh']).agg(n=('correct','size'),oa=('correct','mean'),base_oa=('base_correct','mean'),
        rescue=('rescue','sum'),harm=('harm','sum')).to_csv(ROOT/'geometry_availability_effect.csv')

    # Average outcomes over seeds before cluster resampling; not 3N independent chips.
    a = p[p.arm.eq('late_both')].groupby('obs_id').agg(correct=('correct', 'mean'), mmsi=('mmsi', 'first'), class_id=('class_id', 'first'), base_prediction=('base_prediction', 'first'))
    a['delta'] = a.correct - a.base_prediction.eq(a.class_id)
    grouped = a.groupby(['mmsi', 'class_id']).agg(n=('correct', 'size'), delta=('delta', 'sum'))
    ships = sorted(a.mmsi.unique()); sid = {s:i for i,s in enumerate(ships)}
    n = np.zeros((len(ships), 8)); delta = np.zeros_like(n)
    for (ship, c), row in grouped.iterrows(): n[sid[ship], c] = row.n; delta[sid[ship], c] = row.delta
    rng = np.random.default_rng(20261020); boot = []
    for _ in range(500):
        ii = rng.integers(len(ships), size=len(ships)); nn = n[ii].sum(0); dd = delta[ii].sum(0)
        boot.append([dd.sum()/nn.sum(), np.mean(dd[nn>0]/nn[nn>0])])
    pd.DataFrame([dict(metric=k, lo=float(np.quantile(np.array(boot)[:, j], .025)), hi=float(np.quantile(np.array(boot)[:, j], .975))) for j,k in enumerate(('oa', 'ba'))]).to_csv(ROOT / 'conditional_ci.csv', index=False)

    # Cases use the predeclared late_both, seed0, not the best test arm or seed.
    a = p[p.arm.eq('late_both') & p.seed.eq(0)].set_index('obs_id')
    base = pd.read_csv(AXIS / 'predictions.csv').query('arm == "axis"').set_index('obs_id').loc[a.index]
    diff = a.margin - base.margin
    chosen = []
    for name, mask, ascending in [('rescue', a.correct & ~base.correct, False), ('harm', ~a.correct & base.correct, True),
                                  ('persistent_correct', a.correct & base.correct, False), ('persistent_wrong', ~a.correct & ~base.correct, True)]:
        for oid in diff[mask].sort_values(ascending=ascending).index[:2 if name in ('rescue', 'harm') else 1]: chosen.append((name, int(oid)))
    # Posthoc mechanism illustrations for the two aggregate-benefit classes,
    # explicitly restricted to available VV/VH radar references. Not selection.
    if ROOT != SOURCE:
        for cls in (3,4):
            mask=a.correct & ~base.correct & a.class_id.eq(cls) & available.loc[a.index,'radar_vh'].eq(1)
            for oid in diff[mask].sort_values(ascending=False).index[:1]:
                if int(oid) not in [v for _,v in chosen]: chosen.append((f'rescue_radar_available_c{cls}',int(oid)))
    views = {s['obs_id']:s for s in json.loads((LOCAL / 'view_snapshot.json').read_text(encoding='utf-8'))}
    meta = f.set_index('obs_id'); position = {oid:i for i,oid in enumerate(f.obs_id)}
    names = f.drop_duplicates('class_id').set_index('class_id').class_name.to_dict()
    cache = np.load(SOURCE / 'late_both.npy', mmap_mode='r')
    conflict = pd.read_csv(LOCAL / 'crossview_alignment.csv').set_index('obs_id')
    records = []; contributions = []
    for outcome, oid in chosen:
        row = meta.loc[oid]; result = a.loc[oid]; b = base.loc[oid]; fold = int(result.fold)
        head = torch.load(ROOT / f'head_f{fold}_late_both_s0.pt', map_location='cpu', weights_only=False)
        net = Residual(); net.load_state_dict(head['state']); net.eval()
        raw = np.asarray(cache[position[oid]]).reshape(60,64)
        z = torch.from_numpy((raw-head['mean'])/head['std'])
        with torch.inference_mode():
            hidden = F.gelu(net.content(z)).numpy()
            empty_hidden = F.gelu(net.content(torch.from_numpy(-head['mean']/head['std']))).numpy()
        logits = hidden[None] * net.output.weight.detach().numpy().reshape(8,60,8)
        slot = logits.sum(-1)
        empty_slot=(empty_hidden[None]*net.output.weight.detach().numpy().reshape(8,60,8)).sum(-1)
        score = np.load(ROOT / f'scores_f{fold}_late_both_s0.npz')
        j = int(np.flatnonzero(score['rows']==position[oid])[0]); yy = int(row.class_id)
        rival = int(b.prediction) if int(b.prediction) != yy else int(np.argmax(np.where(np.arange(8)==yy,-np.inf,score['base'][j])))
        terms = (slot[yy]-slot[rival]).reshape(2,3,2,5)
        empty_terms=(empty_slot[yy]-empty_slot[rival]).reshape(2,3,2,5)
        image_terms=terms-empty_terms
        bias = float(net.output.bias.detach()[yy]-net.output.bias.detach()[rival])
        empty_bias=bias+float(empty_terms.sum())
        base_margin = float(score['base'][j,yy]-score['base'][j,rival])
        new_margin = float(score['scores'][j,yy]-score['scores'][j,rival])
        assert np.isclose(base_margin+terms.sum()+bias,new_margin,atol=2e-5)
        assert np.isclose(base_margin+image_terms.sum()+empty_bias,new_margin,atol=2e-5)
        for pol in range(2):
            for role in range(3):
                for ref in range(2):
                    contributions.append(dict(obs_id=oid,seed=0,fold=fold,true_class=yy,rival=rival,pol=('vv','vh')[pol],
                        role=('target','other','rest')[role],reference=('ship_or_image_fallback','radar_or_image_fallback')[ref],
                        term=float(terms[pol,role,ref].sum()),empty_term=float(empty_terms[pol,role,ref].sum()),
                        descriptor_deviation_term=float(image_terms[pol,role,ref].sum())))
        fig, axs = plt.subplots(2,2,figsize=(10,9))
        fig.suptitle(f'{outcome} obs {oid} | {row.class_name} | {row.port}\nfrozen {names[int(b.prediction)]} -> late {names[int(result.prediction)]}, seed0 step{int(result.step)}')
        for pj, pol in enumerate(('vv','vh')):
            im = np.asarray(Image.open(row['image_'+pol]).convert('L')); view=views[oid]['images'][pol]
            m = old.masks_for(view); g, radii = geometry(view)
            for rr in range(2): axs[rr,pj].imshow(im,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            disabled = bool(pj == 1 and conflict.loc[oid,'disable_new_vh'])
            axs[0,pj].set_title(f'{pol.upper()} original {im.shape[1]}x{im.shape[0]}' + (' DISABLED' if disabled else ''))
            for role,color in ((0,'lime'),(1,'orange')):
                if m[role].any() and (~m[role].astype(bool)).any(): axs[1,pj].contour(m[role],levels=[.5],colors=[color],linewidths=.8)
            axis=view['priors'].get('ship_axis')
            if axis:
                cx,cy=axis['center']
                for angle,color,valid in ((g[0],'cyan',g[2]),(g[1],'magenta',g[3])):
                    for theta,radius in ((angle,4*radii[0]),(angle+np.pi/2,4*radii[1])):
                        axs[1,pj].plot([cx-radius*np.cos(theta),cx+radius*np.cos(theta)], [cy-radius*np.sin(theta),cy+radius*np.sin(theta)],
                            'o-' if valid else 'o--',color=color if valid else 'gray',markersize=3,linewidth=.7)
            axs[1,pj].set_title('T/O masks; sampling schematic at OBB center\n'+('radar valid' if g[3] else 'radar missing: image-axis fallback'))
            axs[1,pj].set_xlim(-.5,im.shape[1]-.5); axs[1,pj].set_ylim(im.shape[0]-.5,-.5)
            records.append(dict(outcome=outcome,obs_id=oid,pol=pol,path=row['image_'+pol],width=im.shape[1],height=im.shape[0],
                vh_disabled=disabled,fold=fold,seed=0,step=int(result.step),base_prediction=int(b.prediction),prediction=int(result.prediction),
                true_class=yy,rival=rival,base_fixed_rival_margin=base_margin,new_fixed_rival_margin=new_margin,residual_bias=bias,
                empty_effective_bias=empty_bias,pol_term=float(terms[pj].sum()),pol_descriptor_deviation=float(image_terms[pj].sum()),
                ship_valid=g[2],radar_valid=g[3],target_pixels=float(m[0].sum()),other_pixels=float(m[1].sum())))
        fig.tight_layout(); fig.savefig(ROOT / f'case_{outcome}_{oid}.png',dpi=130); plt.close(fig)
    pd.DataFrame(records).to_csv(ROOT / 'case_review.csv',index=False)
    pd.DataFrame(contributions).to_csv(ROOT / 'case_contributions.csv',index=False)
    print(per_seed.groupby('arm')[['oa','ba','brier']].agg(['mean','std']).to_string())
    print('selected cases',chosen)
    print('selected steps',p[['fold','arm','seed','step']].drop_duplicates().groupby(['arm','step']).size().to_string())


if __name__ == '__main__':
    if len(sys.argv) > 1:
        assert sys.argv[1] == 'mse'
        ROOT = ROOT / 'mse_aligned'
    main()
