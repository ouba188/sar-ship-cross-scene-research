"""Fixed-model common/contrast gain decomposition; no fitting or training."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import sys
sys.dont_write_bytecode = True
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
NEW = PARENT/'partition_replication'
sys.path.insert(0,str(PARENT))
from run_anchored import EXP, STRONG, Anchored, KINDS, SEEDS, weights, metrics
sys.path.insert(0,str(NEW))
from analyze_replication import intervals


def channel_groups(names):
    output = []
    for name in names:
        if name == 'stem':
            groups = [np.r_[p*128+np.arange(g*8,(g+1)*8),p*128+64+np.arange(g*8,(g+1)*8)] for p in range(2) for g in range(8)]
            start = 0
        else:
            width = 2048 if name == 'layer3' else 4096
            groups = np.array_split(np.arange(width),16)
            start = 16
        output.extend((start+g,name,cols) for g,cols in enumerate(groups))
    return output


def weighted_anchor(cohort,fold,d,raw,t,names):
    path = (PARENT/'centering_probe' if cohort=='old' else NEW)/'origins'/f'fold{fold}_outer.npz'
    with np.load(path) as f:
        o = {k:f[k] for k in f.files}
    train,test = d['train'],d['test']
    np.testing.assert_array_equal(o['fit'],train)
    if cohort == 'old':
        np.testing.assert_array_equal(o['selected_names'],names)
    y = t.class_id.to_numpy()
    sw = weights(y[train],t.mmsi.to_numpy()[train])
    target_mean = np.average(np.eye(3)[y[train]],axis=0,weights=sw)*3
    b = d['bias'] if cohort=='old' else o['bias']
    w,centers,weighted = {},{},{}
    for name in names:
        w[name] = o[name+'_W'] if cohort=='old' else o['W']
        centers[name] = o[name+'_h0'] if cohort=='old' else o['h0']
        np.testing.assert_allclose(centers[name],raw[name][train].mean(0,dtype=np.float64),atol=1e-12,rtol=0)
        weighted[name] = np.average(raw[name][train].astype(np.float64),axis=0,weights=sw)
    shift = np.zeros((32,3),np.float64)
    reconstructed = np.zeros_like(d['q'][test])
    for g,name,cols in channel_groups(names):
        shift[g] = w[name][:,cols] @ (weighted[name][cols]-centers[name][cols])
        reconstructed[:,g] = (raw[name][test][:,cols]-centers[name][cols]) @ w[name][:,cols].T
    q_error = float(abs(reconstructed-d['q'][test]).max())
    assert q_error < 1e-7
    bw = b+shift.sum(0)
    weighted_scores = np.average(d['base'][train],axis=0,weights=sw)
    direct = np.broadcast_to(o['raw_intercept'],(len(test),3)).copy()
    for name in names:
        direct += raw[name][test].astype(np.float64) @ w[name].T
    direct_error = float(abs(direct-d['base'][test]).max())
    assert direct_error < 3e-4
    np.testing.assert_array_equal(direct.argmax(1),d['base'][test].argmax(1))
    assert abs(bw-target_mean).max() < 3e-4
    details = dict(cohort=cohort,fold=fold,selected_kind='_'.join(names),source_n=len(train),groups=len(channel_groups(names)),
        q_restore_error=q_error,baseline_restore_error=direct_error,
        weighted_anchor_error=float(abs(bw-target_mean).max()),weighted_score_error=float(abs(weighted_scores-target_mean).max()),
        **{f'b{c}':float(b[c]) for c in range(3)},**{f'bw{c}':float(bw[c]) for c in range(3)},
        **{f'weighted_label_score{c}':float(target_mean[c]) for c in range(3)})
    np.savez_compressed(HERE/f'{cohort}_fold{fold}_anchors.npz',train=train,group_shift=shift,b=b,bw=bw,
        **{name+'_h0w':weighted[name] for name in names})
    return b,bw,shift,details


def render_cases(t,p,stats):
    cases,panels = [],[]
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
    for cohort in ['old','new']:
        original = p[p.cohort.eq(cohort)&p.seed.eq(29)&p.variant.eq('original')]
        for outcome in ['rescued','harmed','CC','WW']:
            subset = original[original.outcome.eq(outcome)&original.neighbors.eq(0)].sort_values('instance_id')
            if subset.empty:
                subset = original[original.outcome.eq(outcome)].sort_values('instance_id')
            if subset.empty:
                continue
            selected = subset.iloc[0]
            iid = selected.instance_id
            r = t.set_index('instance_id').loc[iid]
            part = p[p.cohort.eq(cohort)&p.seed.eq(29)&p.instance_id.eq(iid)]
            diag = stats[stats.cohort.eq(cohort)&stats.seed.eq(29)&stats.instance_id.eq(iid)].iloc[0]
            ims = [Image.open(EXP/'readout/native_targets'/f'{iid}_{pol}.png').convert('RGB') for pol in ['vv','vh']]
            canvas = Image.new('RGB',(1100,245+max(im.height for im in ims)),'white')
            ink = ImageDraw.Draw(canvas)
            ink.text((10,8),f'{cohort} {iid} {outcome}; seed29 fold{selected.fold}; truth={r.class_name}; {r.port}',font=font,fill='black')
            ink.text((10,31),f'Native pixels, VV/VH separate; gain mean={diag.gain_mean:.6f} std={diag.gain_std:.6f}; groups={diag.groups}',font=font,fill='black')
            ink.text((10,54),f'A pred={selected.base_prediction}; true margin={diag.base_margin:+.6f}',font=font,fill='black')
            for line,rr in enumerate(part.itertuples()):
                ink.text((10,77+23*line),f'{rr.variant}: pred={rr.prediction}; true margin={rr.margin:+.6f}',font=font,fill='black')
                cases.append(dict(cohort=cohort,instance_id=iid,outcome=outcome,seed=29,fold=selected.fold,variant=rr.variant,
                    truth=int(r.class_id),prediction=rr.prediction,base_prediction=rr.base_prediction,margin=rr.margin,
                    base_margin=diag.base_margin,gain_mean=diag.gain_mean,gain_std=diag.gain_std,
                    b_true=diag[f'b{int(r.class_id)}'],b_base=diag[f'b{int(selected.base_prediction)}'],
                    port=r.port,pass_group=r.pass_group,native_vv=str(EXP/'readout/native_targets'/f'{iid}_vv.png'),
                    native_vh=str(EXP/'readout/native_targets'/f'{iid}_vh.png'),window_vv=r.target_window_vv,window_vh=r.target_window_vh,
                    selection='seed29 first ID per original outcome; prefer no external neighbors'))
            for x,pol,im in zip([10,560],['VV','VH'],ims):
                ink.text((x,160),f'{pol} {im.width}x{im.height}',font=font,fill='black')
                canvas.paste(im,(x,184))
            path = HERE/f'case_{cohort}_{iid}.png'
            canvas.save(path)
            panels.append(str(path))
    pd.DataFrame(cases).to_csv(HERE/'review_cases.csv',index=False)
    return panels


def main():
    started = time.perf_counter()
    torch.set_num_threads(2)
    t = pd.read_csv(EXP/'readout/targets.csv',dtype={'mmsi':str})
    with np.load(EXP/'readout/features.npz') as f:
        raw = {'stem':f['target']}
        np.testing.assert_array_equal(f['instance_ids'],t.instance_id)
    with np.load(STRONG/'deep_features.npz') as f:
        raw['layer3'] = f['layer3']
        np.testing.assert_array_equal(f['instance_ids'],t.instance_id)
    predictions,statrows,checks,anchors,foldmetrics = [],[],[],[],[]
    for cohort in ['old','new']:
        folder = PARENT if cohort=='old' else NEW
        arm = 'own_regions_full' if cohort=='old' else 'centered_own_full'
        saved = pd.read_csv(folder/('control_predictions.csv' if cohort=='old' else 'predictions.csv'))
        saved = saved[saved.arm.eq(arm)]
        for fold in range(6):
            with np.load(folder/f'fold{fold}_inputs.npz') as f:
                d = {k:f[k] for k in ['base','q','train','test']}
                if cohort=='old':
                    d['bias'] = f['bias']
                    names = KINDS[str(f['selected_kind'])]
                else:
                    names = ['layer3']
                    env = f['env']
            if cohort=='old':
                with np.load(PARENT/f'fold{fold}_own_input.npz') as f:
                    env = f['env']
            active = np.array([g for g,_,_ in channel_groups(names)])
            inactive = np.setdiff1d(np.arange(32),active)
            assert not d['q'][:,inactive].any()
            b,bw,shift,ac = weighted_anchor(cohort,fold,d,raw,t,names)
            anchors.append(ac)
            test = d['test']
            q = d['q'][test]
            base = d['base'][test]
            tq = torch.tensor(q,dtype=torch.float32)
            te = torch.tensor(env[test],dtype=torch.float32)
            valid = torch.ones(len(test))
            y = t.class_id.to_numpy()[test]
            for seed in SEEDS:
                model = Anchored().eval()
                model.load_state_dict(torch.load(folder/'models'/f'fold{fold}_{arm}_{seed}.pt',map_location='cpu',weights_only=True))
                with torch.inference_mode():
                    a = (.5*torch.tanh(model.net(te))).numpy().astype(np.float64)
                    delta_torch = model(te,tq,valid).numpy()
                mean = a[:,active].mean(1)
                contrast = a[:,active]-mean[:,None]
                common_delta = mean[:,None]*q[:,active].sum(1)
                contrast_delta = (contrast[:,:,None]*q[:,active]).sum(1)
                original_delta = (a[:,active,None]*q[:,active]).sum(1)
                split_error = float(abs(common_delta+contrast_delta-original_delta).max())
                assert split_error < 1e-12
                original = base+original_delta
                previous = saved[saved.fold.eq(fold)&saved.seed.eq(seed)].set_index('instance_id').loc[t.iloc[test].instance_id]
                replay_error = float(abs(original-previous[['s0','s1','s2']].to_numpy()).max())
                assert replay_error < 2e-5
                np.testing.assert_array_equal(original.argmax(1),previous.prediction)
                neutral = base+mean[:,None]*(q[:,active]-shift[active]).sum(1)
                neutral_formula = (1+mean[:,None])*base-mean[:,None]*bw
                neutral_error = float(abs(neutral-neutral_formula).max())
                neutral_changes = int((neutral.argmax(1)!=base.argmax(1)).sum())
                assert neutral_changes==0 and neutral_error < 2e-5 and (1+mean>0).all()
                formula_error = float(abs(base+common_delta-((1+mean[:,None])*base-mean[:,None]*b)).max())
                checks.append(dict(cohort=cohort,fold=fold,seed=seed,groups=len(active),
                    split_error=split_error,replay_error=replay_error,torch_error=float(abs(original_delta-delta_torch).max()),
                    common_formula_error=formula_error,neutral_formula_error=neutral_error,neutral_argmax_changes=neutral_changes))
                variants = {'original':original,'common_only':base+common_delta,'contrast_only':base+contrast_delta}
                bp = base.argmax(1)
                for variant,score in variants.items():
                    pred = score.argmax(1)
                    net = (pred==y).astype(int)-(bp==y).astype(int)
                    foldmetrics.append(dict(cohort=cohort,fold=fold,seed=seed,variant=variant,n=len(test),
                        rescued=int((net==1).sum()),harmed=int((net==-1).sum()),net=int(net.sum()),**metrics(y,score)))
                    for j,i in enumerate(test):
                        predictions.append(dict(cohort=cohort,fold=fold,seed=seed,variant=variant,instance_id=t.iloc[i].instance_id,
                            class_id=int(y[j]),base_prediction=int(bp[j]),prediction=int(pred[j]),net=int(net[j]),
                            margin=float(score[j,y[j]]-np.delete(score[j],y[j]).max()),
                            **{f's{c}':float(score[j,c]) for c in range(3)}))
                for j,i in enumerate(test):
                    statrows.append(dict(cohort=cohort,fold=fold,seed=seed,instance_id=t.iloc[i].instance_id,groups=len(active),
                        class_id=int(y[j]),a_mean=float(mean[j]),gain_mean=float(1+mean[j]),gain_std=float(a[j,active].std()),
                        gain_min=float(1+a[j,active].min()),gain_max=float(1+a[j,active].max()),
                        common_norm=float(np.linalg.norm(common_delta[j])),contrast_norm=float(np.linalg.norm(contrast_delta[j])),
                        neutral_prediction=int(neutral[j].argmax()),base_prediction=int(bp[j]),
                        base_margin=float(base[j,y[j]]-np.delete(base[j],y[j]).max()),
                        **{f'b{c}':float(b[c]) for c in range(3)},**{f'base{c}':float(base[j,c]) for c in range(3)}))
                print(cohort,fold,seed,'replay',replay_error,'neutral flips',neutral_changes,flush=True)
    p = pd.DataFrame(predictions).merge(t[['instance_id','neighbors','mmsi','pass_group','port']],on='instance_id',validate='many_to_one')
    p['correct'] = p.prediction.eq(p.class_id)
    p['outcome'] = np.select([p.net.eq(1),p.net.eq(-1),p.correct],['rescued','harmed','CC'],default='WW')
    assert not p.duplicated(['cohort','seed','variant','instance_id']).any()
    assert p.groupby(['cohort','seed','variant']).size().eq(len(t)).all()
    stats = pd.DataFrame(statrows)
    p.to_csv(HERE/'predictions.csv',index=False)
    stats.to_csv(HERE/'sample_gains.csv',index=False)
    pd.DataFrame(anchors).to_csv(HERE/'anchor_verification.csv',index=False)
    pd.DataFrame(checks).to_csv(HERE/'verification.csv',index=False)
    pd.DataFrame(foldmetrics).to_csv(HERE/'fold_metrics.csv',index=False)
    pooled = []
    for (cohort,seed,variant),part in p.groupby(['cohort','seed','variant']):
        pooled.append(dict(cohort=cohort,seed=seed,variant=variant,n=len(part),net=int(part.net.sum()),
            rescued=int(part.net.eq(1).sum()),harmed=int(part.net.eq(-1).sum()),**metrics(part.class_id.to_numpy(),part[['s0','s1','s2']].to_numpy())))
    pd.DataFrame(pooled).to_csv(HERE/'pooled.csv',index=False)
    p.groupby(['cohort','seed','variant','class_id']).agg(n=('instance_id','size'),net=('net','sum'),
        rescued=('net',lambda x:int(x.eq(1).sum())),harmed=('net',lambda x:int(x.eq(-1).sum()))).reset_index().to_csv(HERE/'class_effects.csv',index=False)
    stats.groupby(['cohort','seed']).agg(gain_mean=('gain_mean','mean'),gain_between_sample_sd=('gain_mean','std'),
        gain_within_group_sd_mean=('gain_std','mean'),common_norm_mean=('common_norm','mean'),contrast_norm_mean=('contrast_norm','mean')).reset_index().to_csv(HERE/'gain_summary.csv',index=False)
    ci,paired = [],[]
    for cohort in ['old','new']:
        for variant,reference in [('original','A'),('common_only','A'),('contrast_only','A'),('common_only','original'),('contrast_only','original')]:
            differences = []
            for seed in SEEDS:
                aa = p[p.cohort.eq(cohort)&p.seed.eq(seed)&p.variant.eq(variant)].set_index('instance_id').loc[t.instance_id]
                if reference=='A':
                    ref = aa.base_prediction.to_numpy()==aa.class_id.to_numpy()
                else:
                    ref = p[p.cohort.eq(cohort)&p.seed.eq(seed)&p.variant.eq(reference)].set_index('instance_id').loc[t.instance_id].correct.to_numpy()
                diff = aa.correct.to_numpy().astype(int)-ref.astype(int)
                differences.append(diff)
                paired.append(dict(cohort=cohort,variant=variant,reference=reference,seed=seed,net=int(diff.sum()),
                    OA_delta=float(diff.mean()*100),BA_delta=float(np.mean([diff[t.class_id.eq(c)].mean() for c in range(3)])*100)))
            for block in ['pass_group','mmsi']:
                ci.append(dict(cohort=cohort,variant=variant,reference=reference,**intervals(t,np.mean(differences,axis=0),block)))
    pd.DataFrame(paired).to_csv(HERE/'paired_metrics.csv',index=False)
    pd.DataFrame(ci).to_csv(HERE/'paired_intervals.csv',index=False)
    panels = render_cases(t,p,stats)
    summary = dict(seconds=time.perf_counter()-started,heads_replayed=36,trained_models=0,outer_fits_loaded=12,
        sample_seed_pairs=len(stats),neutral_common_argmax_changes=sum(x['neutral_argmax_changes'] for x in checks),
        max_replay_error=max(x['replay_error'] for x in checks),max_decomposition_error=max(x['split_error'] for x in checks),
        max_weighted_anchor_error=max(x['weighted_anchor_error'] for x in anchors),panels=panels,
        note='Fixed-model diagnostic interventions, not retrained ablations or independent data. All original A scores unchanged.')
    (HERE/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(pd.DataFrame(pooled).to_string(index=False),flush=True)
    print(pd.DataFrame(anchors).to_string(index=False),flush=True)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
