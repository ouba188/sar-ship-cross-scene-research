"""Read-only source review of nine saved heads; no fit or feature extraction."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PROBE = HERE.parent
ROOT = PROBE.parent
sys.path.insert(0, str(PROBE))
import run_topology as run


def paired(frame, reference):
    left = frame[frame.arm.eq('local')].sort_values('source_row').reset_index(drop=True)
    right = left if reference == 'B' else frame[frame.arm.eq(reference)].sort_values('source_row').reset_index(drop=True)
    np.testing.assert_array_equal(left.instance_id, right.instance_id)
    y = left.class_id.to_numpy()
    a = left[[f's{k}' for k in range(8)]].to_numpy()
    b = right[[f'{"b" if reference == "B" else "s"}{k}' for k in range(8)]].to_numpy()
    good, previous = a.argmax(1) == y, b.argmax(1) == y
    ma, mb = run.P.assessment(y, a), run.P.assessment(y, b)
    return dict(n=len(y), rescued=int((good & ~previous).sum()),
                harmed=int((~good & previous).sum()), net=int(good.sum()-previous.sum()),
                top1_changes=int((a.argmax(1) != b.argmax(1)).sum()),
                **{f'delta_{k}':ma[k]-mb[k] for k in ['OA','BA','NLL','balanced_NLL','Brier']})


def verify(predictions, cache, features, table, mapping):
    tr, va = cache['train'], cache['validation']
    audit = []
    count = np.zeros(len(table), int)
    for tag in ['outer'] + [f'inner{k}' for k in range(4)]:
        with np.load(ROOT/'readers/models'/f'fold0_{tag}.npz') as z:
            fit = z['fit']
            held = mapping[va] if tag == 'outer' else z['held']
        a, b = table.set_index('global_row').loc[fit], table.set_index('global_row').loc[held]
        assert set(fit) <= set(mapping[tr])
        for key in ['mmsi','pass_group']:
            assert not set(a[key]) & set(b[key])
        if tag != 'outer':
            ix = np.flatnonzero(np.isin(mapping, held)); count[ix] += 1
            assert (cache['reader'][ix] == int(tag[-1])+1).all()
        audit.append(dict(reader=tag,fit_n=len(fit),held_n=len(held),mmsi_overlap=0,pass_overlap=0))
    assert (count[tr] == 1).all() and (count[va] == 0).all()
    with np.load(ROOT/'features.npz') as z:
        assert z['valid'][mapping].all()
    ratio = features['ratios']; kish = features['kish']
    expected = np.broadcast_to((1/(1-1/kish))[...,None], ratio[...,0].shape)
    complete_error = float(abs(expected-ratio[...,0]).max())
    assert complete_error < 1e-12 and not features['zero_variance'].any()
    point_errors = []
    with np.load(PROBE/'onepoint_replay.npz') as z:
        stems = sorted(k[:-2] for k in z.files if k.endswith('_r'))
        for stem in stems:
            r, wt, saved = z[stem+'_r'],z[stem+'_weight'],z[stem+'_R']
            for pol in range(2):
                w = wt[pol].ravel(); ids = np.flatnonzero(w>0); x = r[pol]
                p = w/w.sum(); variance = ((x-(x*p).sum(1,keepdims=True))**2*p).sum(1)
                ii,jj = np.triu_indices(len(ids),1); i,j = ids[ii],ids[jj]
                edge = (abs(i-j)==14) | ((abs(i-j)==1)&(i//14==j//14))
                for slot,keep in enumerate([np.ones(len(i),bool),edge]):
                    aa,bb=i[keep],j[keep]; weight=w[aa]*w[bb]
                    energy=((x[:,aa]-x[:,bb])**2*weight).sum(1)/(2*weight.sum())
                    err=float(abs(energy/variance-saved[pol,:,slot]).max())
                    assert err<1e-12
                    point_errors.append(err)
        for row in sorted({s.split('_rot')[0] for s in stems}):
            i=int(row[3:]); avg=np.mean([z[f'{row}_rot{k}_R'] for k in range(4)],axis=0)
            np.testing.assert_allclose(avg,ratio[i],atol=1e-12,rtol=0)
    replay=[]
    for seed in run.SEEDS:
        base=run.base_for(seed,cache,mapping)
        initial=None; common=None
        for arm in run.ARMS:
            rr=np.zeros_like(ratio[...,0]) if arm=='marginal' else ratio[...,run.ARMS.index(arm)-1]
            raw=run.descriptor(base,cache,kish,rr)
            if common is None:common=raw[...,:17]
            else:np.testing.assert_array_equal(raw[...,:17],common)
            ck,model=run.restore(PROBE/'heads'/f'{arm}_{seed}.pt')
            np.testing.assert_allclose(ck['mu'],raw[tr].mean((0,1),keepdims=True),atol=0,rtol=0)
            np.testing.assert_allclose(ck['scale'],raw[tr].std((0,1),keepdims=True).clip(1e-8),atol=0,rtol=0)
            x=torch.tensor((raw-ck['mu'])/ck['scale'],dtype=torch.float32)
            with torch.no_grad():score=base+model(x).numpy().astype(float)
            g=predictions[predictions.seed.eq(seed)&predictions.arm.eq(arm)].sort_values('source_row')
            np.testing.assert_array_equal(g.source_row,np.arange(len(table)))
            be=float(abs(g[[f'b{k}' for k in range(8)]].to_numpy()-base).max())
            se=float(abs(g[[f's{k}' for k in range(8)]].to_numpy()-score).max())
            assert max(be,se)<1e-6
            if ck['epoch']==0:np.testing.assert_array_equal(score,base)
            torch.manual_seed(seed);zero=run.Head()
            if initial is None:initial={k:v.clone() for k,v in zero.state_dict().items()}
            else:
                for k,v in zero.state_dict().items():assert torch.equal(v,initial[k])
            with torch.no_grad():assert not zero(x).any()
            replay.append(dict(seed=seed,arm=arm,epoch=ck['epoch'],baseline_error=be,score_error=se,
                               scaler_source_exact=True,common17_exact=True,zero_init_exact=True))
    pd.DataFrame(audit).to_csv(HERE/'source_scope.csv',index=False)
    pd.DataFrame(replay).to_csv(HERE/'replay_checks.csv',index=False)
    return dict(complete_ratio_all_rows_error=complete_error,explicit_ratio_max_error=max(point_errors),
                explicit_ratio_checks=len(point_errors),zero_variance_views=int(features['zero_variance'].sum()),
                valid_all_source_pols=True,source_rows=len(table),training_runs=0,test_evaluated=False)


def render_extra(predictions,features):
    rows=[]
    for outcome in ['rescued','harmed']:
        g=predictions.query("role=='validation' and arm=='local' and seed==29")
        g=g[g.outcome.eq(outcome)].sort_values('instance_id')
        if g.empty:
            rows.append(dict(outcome=outcome,available=False));continue
        row=g.iloc[0];i=int(row.source_row);c=int(row.class_id)
        heads=predictions[(predictions.instance_id==row.instance_id)&predictions.seed.eq(29)].set_index('arm')
        fig,axes=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
        for ax,pol in zip(axes[:2],['vv','vh']):
            with Image.open(row[f'chip_path_{pol}']) as im:image=np.asarray(im)
            ax.imshow(image,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            ax.set_title(f'{pol.upper()} original {image.shape[1]}x{image.shape[0]}')
        b=row[[f'b{k}' for k in range(8)]].to_numpy(float);xx=np.arange(8)
        for j,(name,score) in enumerate([('B',b)]+[(a,heads.loc[a,[f's{k}' for k in range(8)]].to_numpy(float)) for a in run.ARMS]):
            axes[2].bar(xx+(j-1.5)*.2,score,.2,label=name)
        axes[2].set_xticks(xx);axes[2].legend(fontsize=8);axes[2].set_title('Actual class scores')
        rv=b.copy();rv[c]=-np.inf;rival=int(rv.argmax())
        ratio=features['ratios'][i,:,c,1]
        fig.suptitle(f'SOURCE val seed29: {row.instance_id}, true={c}; {outcome} vs B\n'
                     f'B {int(row.base_prediction)} -> local {int(row.prediction)}; all three arms same top1, epoch5\n'
                     f'True-class local ratio VV/VH={ratio[0]:.4f}/{ratio[1]:.4f}; not unique topology benefit',fontsize=10)
        path=HERE/f'{outcome}.png';fig.savefig(path,dpi=115);plt.close(fig)
        rows.append(dict(instance_id=row.instance_id,outcome=outcome,seed=29,epoch=5,available=True,
                         class_id=c,fixed_rival=rival,base_margin=b[c]-b[rival],
                         local_margin=float(row[f's{c}']-row[f's{rival}']),panel=str(path),
                         chip_path_vv=row.chip_path_vv,chip_path_vh=row.chip_path_vh))
    parent=pd.read_csv(PROBE/'cases.csv');parent=parent[parent.available.eq(True)]
    pd.concat([parent,pd.DataFrame(rows)],ignore_index=True).to_csv(HERE/'review_cases.csv',index=False)


def main():
    torch.set_num_threads(2)
    predictions=pd.read_csv(PROBE/'predictions.csv',dtype={'mmsi':str})
    assert set(predictions.role)=={'train','validation'} and set(predictions.fold)=={0}
    with np.load(ROOT/'pair_local_evidence/source_moments.npz') as z:
        cache={k:z[k] for k in ['sample_ids','mean','marginal','reader','train','validation']}
    with np.load(PROBE/'topology_features.npz') as z:features={k:z[k] for k in z.files}
    np.testing.assert_array_equal(features['sample_ids'],cache['sample_ids'])
    allrows=pd.read_csv(ROOT/'cohort.csv',dtype={'mmsi':str})
    mapping=pd.Series(np.arange(len(allrows)),index=allrows.sample_id).loc[cache['sample_ids']].to_numpy()
    table=allrows.iloc[mapping].copy().reset_index(drop=True);table['global_row']=mapping
    checks=verify(predictions,cache,features,table,mapping)
    predictions['support']=np.where(features['kish'][predictions.source_row.to_numpy()].min(1)<4,'neff_lt4','neff_ge4')
    results=[];classes=[];strata=[]
    for (role,seed),g in predictions.groupby(['role','seed']):
        for reference in ['B','marginal','complete']:
            meta=dict(role=role,seed=seed,reference=reference)
            results.append(dict(**meta,**paired(g,reference)))
            for c,h in g.groupby('class_id'):
                classes.append(dict(**meta,class_id=c,**paired(h,reference)))
            for support,h in g.groupby('support'):
                strata.append(dict(**meta,support=support,**paired(h,reference)))
    pd.DataFrame(results).to_csv(HERE/'paired_summary.csv',index=False)
    pd.DataFrame(classes).to_csv(HERE/'class_pairs.csv',index=False)
    pd.DataFrame(strata).rename(columns={'delta_BA':'delta_BA_present_classes'}).to_csv(HERE/'support_pairs.csv',index=False)
    curves=pd.read_csv(PROBE/'selection_curves.csv');selections=pd.read_csv(PROBE/'selections.csv')
    metrics=pd.read_csv(PROBE/'metrics.csv');terminal=[]
    for row in selections.itertuples():
        g=curves[curves.seed.eq(row.seed)&curves.arm.eq(row.arm)].sort_values('epoch')
        # Compare near-equal float J as rational-count ties, not CSV roundoff.
        best=g[g.J>=g.J.max()-1e-14].iloc[0]
        assert int(best.epoch)==row.epoch
        end=g[g.epoch.eq(200)].iloc[0]
        selected=metrics[metrics.seed.eq(row.seed)&metrics.arm.eq(row.arm)&metrics.role.eq('train')].iloc[0]
        terminal.append(dict(seed=row.seed,arm=row.arm,selected_epoch=row.epoch,
            selected_train_jointCE=.5*(selected.NLL+selected.balanced_NLL),
            last_logged_train_loss_before_update200=end.train_loss,
            selected_validation_J=row.validation_J,terminal_validation_J=end.J,
            terminal_validation_OA=end.OA,terminal_validation_BA=end.BA,
            terminal_validation_NLL=end.NLL))
    pd.DataFrame(terminal).to_csv(HERE/'selected_terminal.csv',index=False)
    render_extra(predictions,features)
    checks.update(images_viewed=False,bootstrap_runs=0,expected_panels=4)
    (HERE/'checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(pd.DataFrame(results).query("role=='validation'").to_string(index=False))
    print(pd.DataFrame(terminal).to_string(index=False));print(json.dumps(checks))


if __name__=='__main__':
    main()
