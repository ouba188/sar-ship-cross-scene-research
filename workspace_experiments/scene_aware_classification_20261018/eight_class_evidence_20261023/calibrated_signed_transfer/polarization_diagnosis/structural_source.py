"""Source-only attainable envelopes and four native panels; no training."""
import json
import sys
import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from diagnose_source import HERE, ROOT, ref
sys.path.insert(0, str(ref.PARENT))
from prepare_features import OLD, crop_pad, masks_for


def main():
    torch.set_num_threads(2)
    table = ref.load_table()
    cache = dict(np.load(ROOT/'signed_features.npz'))
    names = list(cache['reader_names'])
    primary = pd.read_csv(HERE/'source_primary.csv', dtype={'mmsi': str})
    frames = []
    for fold in range(3):
        d, outer, oof = ref.load_fold(fold, table, cache)
        op = cache['moments_pol'][names.index(f'fold{fold}_outer')]
        ip = np.zeros_like(op)
        for inner in range(4):
            name = f'fold{fold}_inner{inner}'
            held = np.load(ref.PARENT/'baseline/models'/f'{name}.npz')['held']
            ip[held] = cache['moments_pol'][names.index(name), held]
        b, bt, _ = ref.calibrated(fold, 17, d, outer, oof)
        ck, model = ref.restore(ref.checkpoint_path('increment', fold, 17, 'signed_only'))
        for role, base, pol, mom in [('train_oof', bt, ip, oof), ('validation', b, op, outer)]:
            g = primary[primary.fold.eq(fold)&primary.role.eq(role)].copy()
            ix, y, rival = g.row_index.to_numpy(), g.class_id.to_numpy(), g.fixed_rival.to_numpy()
            n = np.arange(len(g))
            score = base[ix]
            k = pol[ix, :, :, 1].transpose(0, 2, 1)
            u, v = abs(k.sum(-1)), abs(k).sum(-1)
            g['aggregate_pair_bound'] = u[n, y]+u[n, rival]
            g['preserved_pair_bound'] = v[n, y]+v[n, rival]
            g['pair_bound_widened'] = g.preserved_pair_bound.gt(g.aggregate_pair_bound+1e-12)
            g['new_pair_crossable_wrong'] = ~g.B_correct & (-g.B_margin >= g.aggregate_pair_bound) & (-g.B_margin < g.preserved_pair_bound)
            for label, bound in [('aggregate', u), ('preserved', v)]:
                competitors = score-bound
                competitors[n, y] = -np.inf
                margin = score[n, y]+bound[n, y]-competitors.max(1)
                g[label+'_box_margin'] = margin
                g[label+'_box_crossable_wrong'] = ~g.B_correct & (margin > 1e-12)
            g['new_box_crossable_wrong'] = g.preserved_box_crossable_wrong & ~g.aggregate_box_crossable_wrong
            g['true_carrier_opposite'] = k[n,y,0]*k[n,y,1] < 0
            g['rival_carrier_opposite'] = k[n,rival,0]*k[n,rival,1] < 0
            raw = ref.raw_input('signed_only', score, mom[ix])
            with torch.inference_mode():
                gain = model.net(torch.tensor((raw-ck['mu'])/ck['scale'],dtype=torch.float32)).tanh().squeeze(-1).numpy().astype(float)
            parts = gain[:,:,None]*k
            actual = parts[n,y]-parts[n,rival]
            g['actual_vv_margin_delta'], g['actual_vh_margin_delta'] = actual.T
            g['actual_margin_cancel'] = np.where(abs(actual).sum(1)>1e-12,
                1-abs(actual.sum(1))/np.maximum(abs(actual).sum(1),1e-12), np.nan)
            error = abs(actual.sum(1)-(g.S_fixed_margin-g.B_margin)).max()
            assert error < 2e-6
            g['actual_contribution_replay_error'] = error
            frames.append(g)
    result = pd.concat(frames,ignore_index=True)
    result.to_csv(HERE/'source_reachability.csv',index=False)
    rows=[]
    for role,g in result.groupby('role'):
        groupings=[('all','all',g)]
        for col in ['class_id','valid_vh','pass_group','conflict','outcome']:
            groupings += [(col,str(value),h) for value,h in g.groupby(col)]
        for unit,value,h in groupings:
            rows.append(dict(role=role,unit=unit,name=value,n=len(h),B_wrong=int((~h.B_correct).sum()),
                dual_valid=int(h.dual_valid.sum()),conflicts=int(h.conflict.sum()),
                pair_bound_widened=int(h.pair_bound_widened.sum()),
                new_pair_crossable_wrong=int(h.new_pair_crossable_wrong.sum()),
                aggregate_box_crossable_wrong=int(h.aggregate_box_crossable_wrong.sum()),
                preserved_box_crossable_wrong=int(h.preserved_box_crossable_wrong.sum()),
                new_box_crossable_wrong=int(h.new_box_crossable_wrong.sum()),
                raw_opposite_true=int(h.true_carrier_opposite.sum()),raw_opposite_rival=int(h.rival_carrier_opposite.sum())))
    stats=pd.DataFrame(rows)
    stats.to_csv(HERE/'reachability_counts.csv',index=False)
    print(stats[stats.unit.eq('all')].to_string(index=False),flush=True)
    cases=pd.read_csv(HERE/'source_review_cases.csv')
    snapshots={s['obs_id']:s for s in json.loads((OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    panels=[]
    for _,q in cases.iterrows():
        fig,axes=plt.subplots(2,2,figsize=(10,11),layout='constrained')
        dims=[]
        for j,pol in enumerate(['vv','vh']):
            view=snapshots[int(q.obs_id)]['images'][pol]
            with Image.open(q[f'image_{pol}']) as im:
                raw=np.array(im.convert('L'))
            pixels,masks,place=crop_pad(raw,masks_for(view),160)
            dims.append(f'{pol}:{raw.shape[1]}x{raw.shape[0]}')
            axes[0,j].imshow(raw,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            axes[0,j].set_title(f'{pol.upper()} original {raw.shape[1]}x{raw.shape[0]}')
            axes[1,j].imshow(pixels,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            axes[1,j].set_title(f'Actual 160 crop/pad; footprint={masks.sum(0).mean():.3f}')
            prior=view['priors'];target=prior.get('target_index');det=prior.get('detections',[])
            if target is not None and 0<=target<len(det):
                points=np.array(det[target]['corners']);points=np.r_[points,points[:1]]
                axes[0,j].plot(points[:,0],points[:,1],color='#13c74e',lw=1)
                points=points-[place['crop_x'],place['crop_y']]+[place['pad_left'],place['pad_top']]
                axes[1,j].plot(points[:,0],points[:,1],color='#13c74e',lw=1)
            for ax in axes[:,j]:
                ax.set_xlabel('pixel x');ax.set_ylabel('pixel y')
        fig.suptitle(f'SOURCE validation / seed17 / fold{int(q.fold)} / {q.instance_id}\n'
            f'conflict={q.conflict}; B_correct={q.B_correct}; true={int(q.class_id)}; fixed rival={int(q.fixed_rival)}\n'
            f'kappa true-rival: VV {q.d_vv:+.4f}, VH {q.d_vh:+.4f}, sum {q.d_sum:+.4f}; cancel {q.cancel:.3f}\n'
            f'B pred {int(q.B_prediction)} -> signed {int(q.S_prediction)}; epoch {int(q.epoch)}; '
            f'B margin {q.B_margin:+.4f}, final margin {q.S_margin:+.4f}\n'
            'Independent original windows; no pixel registration or multiplicative fusion.',fontsize=10)
        path=HERE/f'source_{q.instance_id}.png';fig.savefig(path,dpi=110);plt.close(fig)
        panels.append(dict(instance_id=q.instance_id,panel=str(path),dimensions=';'.join(dims)))
    pd.DataFrame(panels).to_csv(HERE/'panels.csv',index=False)
    print(pd.DataFrame(panels).to_string(index=False),flush=True)


if __name__=='__main__':
    main()
