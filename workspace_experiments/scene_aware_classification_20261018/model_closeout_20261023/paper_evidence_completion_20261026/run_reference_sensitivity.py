"""Frozen model, alternate source reference banks: sensitivity, not refit success."""
import argparse
import time
import numpy as np
import pandas as pd
from run_classical import HERE, ROOT, TRANSFER, SEEDS, tr, write_json, load_selected
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',choices=('oss1','oss2'),required=True)
    args=p.parse_args()
    name=args.source
    other='oss2' if name=='oss1' else 'oss1'
    direction=f'{name}-to-{other}'
    root=ROOT/f'{name}_m123_replication_20261026'
    cross=TRANSFER/direction
    repair=TRANSFER/'normalization_repair_20261026'
    mroot=root/('m3_results_grid01' if name=='oss1' else 'm3_results')
    table=pd.read_csv(root/'stack_table.csv',dtype={'mmsi':str})
    target=tr.read_table(cross/'target_table.csv')
    test=np.flatnonzero(table.role.eq('evaluation'))
    local=tr.load_npz(mroot/'local.npz')
    protocol=tr.read_json(cross/'protocol.json')
    raw=tr.aligned(Path(protocol['target_cache'])/'m3_raw.npz',target)
    external=tr.project_raw(raw,tr.load_npz(mroot/'projection.npz'))
    del raw
    features={k:np.concatenate((local[k][test],external[k])) for k in tr.FEATURES}
    ids=np.r_[table.iloc[test].mmsi.to_numpy(),target.mmsi.to_numpy()]
    references=tr.load_npz(mroot/'references.npz')
    covariance={k:references[k] for k in ('mu','scale','active')}
    eligible=table.role.eq('fit_train').to_numpy() & (local['mass'][:,:,:8].sum((1,2))>0)
    candidates=table[eligible].drop_duplicates('mmsi')
    out=HERE/'reference_sensitivity'/name
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'protocol.json',dict(draw_seeds=[101,211,307],per_class=8,bank_size=32,
        model_seeds=SEEDS,models_frozen=True,scalers_frozen=True,source_only_bank=True,
        type='frozen deployed-bank sensitivity; not retraining at each reference draw',
        input_projection='same original source PCA',target_labels_used_for_selection=False))
    elapsed=[]
    for draw in (101,211,307):
        rng=np.random.default_rng(draw)
        bank_rows=np.concatenate([rng.choice(candidates.index[candidates.class_id.eq(c)],8,replace=False) for c in range(4)])
        assert table.iloc[bank_rows].mmsi.nunique()==32
        bank={k:local[k][bank_rows] for k in tr.FEATURES}
        refs=tr.retrieve(features,bank,ids,table.iloc[bank_rows].mmsi)
        dest=out/f'draw{draw}'
        dest.mkdir(exist_ok=True)
        table.iloc[bank_rows].to_csv(dest/'bank.csv',index=False)
        for seed in SEEDS:
            source_base=tr.load_npz(repair/f'source_{name}/seed{seed}/predictions.npz')['s12'][test]
            target_base=tr.load_npz(repair/direction/f'seed{seed}/predictions.npz')['s12']
            base=np.r_[source_base,target_base].astype(np.float32)
            tick=time.perf_counter()
            z=tr.m3_predict(features,bank,refs,covariance,base,mroot/f'seed{seed}',device='cuda')
            elapsed.append(dict(draw=draw,seed=seed,n=len(base),seconds=time.perf_counter()-tick))
            np.savez_compressed(dest/f'seed{seed}.npz',**{k:z[k] for k in tr.ARMS},
                refs=bank_rows[refs],bank_rows=bank_rows,within_rows=test,
                within_obs_id=table.iloc[test].obs_id.to_numpy(),cross_obs_id=target.obs_id.to_numpy())
            print('SAVED',name,draw,seed,flush=True)
    target=tr.read_table(cross/'target_table.csv',labels=True)
    rows=[]
    for draw in (101,211,307):
        for seed in SEEDS:
            z=tr.load_npz(out/f'draw{draw}/seed{seed}.npz')
            for cohort,sl,labels in [('within',slice(0,len(test)),table.iloc[test].class_id.to_numpy(int)),
                                    ('cross',slice(len(test),None),target.class_id.to_numpy(int))]:
                original=tr.load_npz(repair/(f'source_{name}' if cohort=='within' else direction)/f'seed{seed}/predictions.npz')
                for arm in tr.ARMS:
                    old=original[arm][test] if cohort=='within' else original[arm]
                    score=z[arm][sl]
                    a,b=score.argmax(1)==labels,old.argmax(1)==labels
                    rows.append(dict(source=name,cohort=cohort,draw=draw,seed=seed,arm=arm,
                        rescued=int((a&~b).sum()),harmed=int((b&~a).sum()),
                        OA_delta_pp=float(100*(a.mean()-b.mean())),
                        BA_delta_pp=float(100*np.mean([a[labels==c].mean()-b[labels==c].mean() for c in range(4)])),
                        **tr.m3.metrics(labels,score)))
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(elapsed).to_csv(out/'timing.csv',index=False)
    write_json(out/'complete.json',dict(draws=3,seeds=3,trained=False,within=len(test),cross=len(target)))
    print(pd.DataFrame(rows).groupby(['cohort','arm'])[['OA','BA','OA_delta_pp','BA_delta_pp']].mean().to_string(),flush=True)


if __name__=='__main__':
    main()
