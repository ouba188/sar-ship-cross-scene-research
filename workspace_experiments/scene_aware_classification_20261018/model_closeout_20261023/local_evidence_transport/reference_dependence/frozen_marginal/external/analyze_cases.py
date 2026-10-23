"""Native paired cases and fixed-coefficient interventions; no fitting."""
import os
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FM = HERE.parent
ROOT = FM.parent.parent
EXT = ROOT / 'external_confirmation'
UP = ROOT.parent / 'regional_utility_pilot/upstream'
OUT = HERE / 'analysis'
sys.path.insert(0, str(ROOT))
from extract_tokens import geo, slots
PAIRS = ((0,1), (0,2), (3,4), (3,5), (0,3), (1,4), (2,5))


def main():
    OUT.mkdir(exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(OUT / '.mplcache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(UP))
    from pipeline import metrics
    table = pd.read_csv(EXT / 'table.csv', dtype={'mmsi':str})
    source = pd.read_csv(UP / 'table.csv', dtype={'mmsi':str})
    tokens = np.load(EXT / 'tokens.npz')
    source_tokens = np.load(ROOT / 'tokens.npz')
    feat = np.load(HERE / 'features.npz')
    weights = np.load(FM / 'weights.npz')
    bank = feat['bank']
    y = table.class_id.to_numpy(int)
    intervention = []
    for seed in (17,29,43):
        pred = np.load(HERE / f'predictions_seed{seed}.npz')
        for arm in ('joint','shuffle101','shuffle211','shuffle307'):
            delta = np.einsum('ncf,f->nc', feat[arm], weights[f'joint_{seed}'])
            delta -= delta.mean(-1, keepdims=True)
            score = pred['anchor'] + delta
            if arm == 'joint':
                np.testing.assert_allclose(score, pred['joint'], atol=2e-6, rtol=0)
            intervention.append(dict(seed=seed, replacement=arm, **metrics(y,score)))
    pd.DataFrame(intervention).to_csv(OUT / 'fixed_joint_head_interventions.csv', index=False)
    pred = np.load(HERE / 'predictions_seed17.npz')
    old = pred['anchor'].argmax(-1) == y
    new = pred['joint'].argmax(-1) == y
    masks = {'rescued': ~old & new, 'harmed': old & ~new,
             'CC': old & new, 'WW': ~old & ~new,
             'fishing_harmed': old & ~new & (y == 1),
             'general_rescued': ~old & new & (y == 2)}
    cases = []
    for name, mask in masks.items():
        candidates = np.flatnonzero(mask)
        if not len(candidates):
            continue
        i = int(candidates[np.argmin(table.iloc[candidates].obs_id.to_numpy())])
        c = int(y[i])
        rival = int(np.argmax(np.where(np.arange(8)==c, -np.inf, pred['joint'][i])))
        parts = weights['joint_17']*(feat['joint'][i,c]-feat['joint'][i,rival])
        pair = int(np.argmax(np.abs(parts)))
        g,h = PAIRS[pair]
        refs = {}
        for cl in (c,rival):
            u = feat['u'][i]
            ids = np.flatnonzero(np.isfinite(u[g,cl]) & np.isfinite(u[h,cl]))
            order = ids[np.argsort(-(u[g,cl,ids]+u[h,cl,ids]), kind='stable')[:2]]
            refs[cl] = bank[cl,order].tolist()
        cases.append(dict(outcome=name,n=int(mask.sum()),i=i,obs_id=int(table.iloc[i].obs_id),true=c,rival=rival,pair=pair,refs=refs))
    wanted = {int(table.iloc[ca['i']].obs_id) for ca in cases}
    wanted |= {int(source.iloc[ix].obs_id) for ca in cases for cl in ca['refs'] for ix in ca['refs'][cl]}
    snaps = {int(s['obs_id']):s for s in json.loads((geo.OLD / 'local_condition_module/view_snapshot.json').read_text(encoding='utf-8')) if int(s['obs_id']) in wanted}
    records, margins, attribution = [],[],[]
    for ca in cases:
        i,c,r = ca['i'],ca['true'],ca['rival']
        fig,axes = plt.subplots(2,5,figsize=(20,10))
        fig.subplots_adjust(top=.83,bottom=.16,wspace=.15,hspace=.30)
        views = [(table,i,tokens,'QUERY')]
        views += [(source,ix,source_tokens,f'TRUE REF {n+1}') for n,ix in enumerate(ca['refs'][c])]
        views += [(source,ix,source_tokens,f'RIVAL REF {n+1}') for n,ix in enumerate(ca['refs'][r])]
        for p,pol in enumerate(('vv','vh')):
            for column,(tab,ix,cache,label) in enumerate(views):
                pix,foot,g,placement = geo.native(tab,snaps,ix,p)
                valid = bool(cache['valid'][ix,p])
                ax = axes[p,column]
                ax.imshow(pix,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                if valid:
                    masks = slots(g,foot)
                    for (lo,hi),color in zip(((0,8),(8,16),(16,20)),('#24a878','#2c84c6','#d39222')):
                        field = masks[lo:hi].sum(0)
                        if field.min() < .5 < field.max():
                            ax.contour(field,levels=[.5],colors=[color],linewidths=.7)
                row = tab.iloc[ix]
                ax.set(title=f'{pol.upper()} {label}: obs {int(row.obs_id)}\n{row.class_name}; {row.port}\nlocal valid={int(valid)}',xlim=(-.5,159.5),ylim=(159.5,-.5),xticks=[0,80,159],yticks=[0,80,159])
                records.append(dict(case=ca['outcome'],query_obs=ca['obs_id'],scope='external' if column==0 else 'source',obs_id=int(row.obs_id),pol=pol,path=row['image_'+pol],valid=valid,placement=placement))
        lines=[]
        base_margin = float(pred['anchor'][i,c]-pred['anchor'][i,r])
        for arm in ('full','anchor','nonlinear','joint','shuffle307'):
            margin=float(pred[arm][i,c]-pred[arm][i,r])
            margins.append(dict(case=ca['outcome'],obs_id=ca['obs_id'],arm=arm,true=c,rival=r,prediction=int(pred[arm][i].argmax()),margin=margin,change_vs_anchor=margin-base_margin))
            lines.append(f'{arm}: pred={int(pred[arm][i].argmax())}; margin={margin:+.4f}; delta={margin-base_margin:+.4f}')
        parts = weights['joint_17']*(feat['joint'][i,c]-feat['joint'][i,r])
        assert abs(parts.sum()-(pred['joint'][i,c]-pred['joint'][i,r]-base_margin))<3e-6
        for f,val in enumerate(parts):
            attribution.append(dict(case=ca['outcome'],obs_id=ca['obs_id'],pair=f,contribution=float(val)))
        fig.suptitle(f"External reference dependence | seed17 | {ca['outcome']} vs anchor | obs {ca['obs_id']} | group n={ca['n']}\n8 references/class; top2 numerator refs displayed for pair {ca['pair']}={PAIRS[ca['pair']]}. Not all causal contributions.",fontsize=12)
        fig.text(.04,.018,'\n'.join(lines)+'\nNative160 crop/pad, no resize. VV/VH separate. Green body / blue side / amber end. Layer1 RF=35, regions not pixel-pure.',fontsize=9)
        fig.savefig(OUT / f"case_{ca['outcome']}_{ca['obs_id']}.png",dpi=130)
        plt.close(fig)
    pd.DataFrame(margins).to_csv(OUT / 'case_margins.csv',index=False)
    pd.DataFrame(attribution).to_csv(OUT / 'head_contributions.csv',index=False)
    (OUT / 'cases.json').write_text(json.dumps(cases,indent=2),encoding='utf-8')
    (OUT / 'image_records.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print(pd.DataFrame(cases)[['outcome','n','obs_id','pair']].to_string(index=False))
    print(pd.DataFrame(margins).to_string(index=False))
    print(pd.DataFrame(intervention).groupby('replacement')[['OA','BA','NLL','Brier']].mean().to_string())


if __name__ == '__main__':
    main()
