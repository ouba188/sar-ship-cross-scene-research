"""Paired fixed-prediction intervals and original query/donor image review."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_diagnosis import HERE, run
sys.path.insert(0, str(run.parent.HERE))
import analyze_results as common


def main():
    p = pd.read_csv(HERE/'predictions.csv', dtype={'mmsi':str, 'donor_mmsi':str})
    ci = []
    for (match, seed), g in p[p.role.eq('test')].groupby(['match','seed']):
        ci.extend(dict(match=match, seed=seed, **r) for r in
            common.intervals(g, g.net_own_vs_borrowed.to_numpy(), 'own-borrowed'))
    pd.DataFrame(ci).to_csv(HERE/'intervals.csv', index=False)
    table = run.parent.load_table().set_index('instance_id')
    snapshots = {s['obs_id']:s for s in json.loads((run.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    (HERE/'panels').mkdir(exist_ok=True)
    chosen = p[p.role.eq('test') & p.seed.eq(17) & p.match.eq('B_and_mean')]
    cases = []
    for outcome in ['rescued','harmed','CC','WW']:
        group = chosen[chosen.outcome.eq(outcome)].sort_values('instance_id')
        if group.empty:
            continue
        q = group.iloc[0]
        fig, axes = plt.subplots(2,3, figsize=(14,9),layout='constrained')
        rec = dict(outcome=outcome, query=q.instance_id, donor=q.donor_id,
            fold=int(q.fold), seed=int(q.seed), epoch=int(q.epoch), true_class=int(q.class_id),
            donor_class=int(q.donor_class), own_prediction=int(q.own_prediction),
            borrowed_prediction=int(q.borrowed_prediction), B_mean_distance=q.B_mean_distance)
        for col, (label, ident) in enumerate([('Query',q.instance_id),('Donor',q.donor_id)]):
            row = table.loc[ident]
            for j, pol in enumerate(['vv','vh']):
                path = row[f'image_{pol}']
                with Image.open(path) as im:
                    raw = np.array(im.convert('L'))
                axes[j,col].imshow(raw,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                axes[j,col].set_title(f'{label} {pol.upper()} {raw.shape[1]}x{raw.shape[0]} px\n{ident} / class {row.class_id}')
                view = snapshots[int(row.obs_id)]['images'][pol]
                prior = view['priors']; target=prior.get('target_index'); det=prior.get('detections',[])
                if target is not None and 0 <= target < len(det):
                    corners = np.array(det[target]['corners']); corners=np.r_[corners,corners[:1]]
                    axes[j,col].plot(corners[:,0],corners[:,1],color='#13b978',lw=1)
                rec[f'{label.lower()}_{pol}_path'] = str(path)
                rec[f'{label.lower()}_{pol}_shape'] = str(raw.shape)
        b = q[[f'b{c}' for c in range(8)]].to_numpy(float)
        own = q[[f'own{c}' for c in range(8)]].to_numpy(float)
        borrow = q[[f'borrowed{c}' for c in range(8)]].to_numpy(float)
        x=np.arange(8)
        for ax in axes[:,2]:
            ax.set_xticks(x); ax.axhline(0,color='black',lw=.5)
        axes[0,2].bar(x-.25,b,.25,label='B unchanged',color='#757575')
        axes[0,2].bar(x,own,.25,label='Own distribution',color='#137e65')
        axes[0,2].bar(x+.25,borrow,.25,label='Borrowed distribution',color='#bd6c1d')
        axes[0,2].legend(fontsize=8);axes[0,2].set_title('Same query B and mean')
        axes[1,2].bar(x,own-borrow,color='#167b9b')
        axes[1,2].set_title('Own minus borrowed score; no physical attribution')
        y=int(q.class_id); rr=borrow.copy();rr[y]=-np.inf;rival=int(rr.argmax())
        rec.update(fixed_rival=rival,own_margin=float(own[y]-own[rival]),
            borrowed_margin=float(borrow[y]-borrow[rival]), base_prediction=int(b.argmax()))
        fig.suptitle(f'{outcome}: own vs borrowed, not vs B / fold{q.fold} seed17 epoch{q.epoch}\n'
            f'query={q.instance_id} / donor={q.donor_id}; different MMSI and pass\n'
            f'No label used in matching; B+mean distance {q.B_mean_distance:.4f}; '
            f'fixed rival {rival} margin {rec["borrowed_margin"]:+.4f} -> {rec["own_margin"]:+.4f}',fontsize=11)
        path=HERE/'panels'/f'{outcome}.png'
        fig.savefig(path,dpi=110);plt.close(fig)
        rec['panel']=str(path);cases.append(rec)
    pd.DataFrame(cases).to_csv(HERE/'review_cases.csv',index=False)
    print(pd.DataFrame(cases)[['outcome','query','donor','true_class','donor_class','epoch','own_margin','borrowed_margin']].to_string(index=False))


if __name__ == '__main__':
    main()
