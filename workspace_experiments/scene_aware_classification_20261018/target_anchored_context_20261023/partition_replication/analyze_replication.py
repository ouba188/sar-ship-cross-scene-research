"""Read-only model replay, paired summaries, and native target review panels."""
import sys
sys.dont_write_bytecode = True
import json
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from run_replication import HERE, EXP, STRONG, ARMS, SEEDS, Anchored, WithBias, metrics


def margin(scores, truth):
    return float(scores[truth]-np.delete(scores,truth).max())


def intervals(t, diff, block):
    labels, ix = np.unique(t[block].to_numpy(str),return_inverse=True)
    sizes = np.bincount(ix)
    sums = np.bincount(ix,weights=diff)
    ys = t.class_id.to_numpy()
    counts = np.stack([np.bincount(ix,weights=(ys==c),minlength=len(labels)) for c in range(3)],axis=1)
    effects = np.stack([np.bincount(ix,weights=diff*(ys==c),minlength=len(labels)) for c in range(3)],axis=1)
    rng = np.random.default_rng(20261023)
    draws = rng.integers(len(labels),size=(2000,len(labels)))
    oa = sums[draws].sum(1)/sizes[draws].sum(1)*100
    ba = (effects[draws].sum(1)/counts[draws].sum(1)).mean(1)*100
    return dict(block=block,blocks=len(labels),draws=2000,seed=20261023,
                OA_low=float(np.quantile(oa,.025)),OA_high=float(np.quantile(oa,.975)),
                BA_low=float(np.quantile(ba,.025)),BA_high=float(np.quantile(ba,.975)))


def scale_diagnostics():
    rows = []
    for fold in range(6):
        with np.load(HERE/f'fold{fold}_inputs.npz') as f:
            train = f['train']
            qc, qr = f['q'][train], f['qraw'][train]
        with np.load(HERE/'origins'/f'fold{fold}_outer.npz') as f:
            origin = f['origin']
        for g in range(16,32):
            centered = float(np.sqrt(np.mean(qc[:,g]**2)))
            raw = float(np.sqrt(np.mean(qr[:,g]**2)))
            off = float(np.sqrt(np.mean(origin[g]**2)))
            rows.append(dict(fold=fold,group=g,source_n=len(train),centered_rms=centered,
                raw_rms=raw,origin_rms=off,origin_over_centered=off/centered))
    frame = pd.DataFrame(rows)
    frame.to_csv(HERE/'scale_diagnostics.csv',index=False)
    print('OUTER SOURCE GROUP SCALES',frame.sort_values('origin_over_centered').tail(5).to_string(index=False),flush=True)


def main():
    torch.set_num_threads(2)
    scale_diagnostics()
    t = pd.read_csv(EXP/'readout/targets.csv',dtype={'mmsi':str})
    p = pd.read_csv(HERE/'predictions.csv')
    p = p.merge(t[['instance_id','mmsi','pass_group','port','neighbors']],on='instance_id',validate='many_to_one')
    p['correct'] = p.prediction.eq(p.class_id)
    p['base_correct'] = p.base_prediction.eq(p.class_id)
    p['net'] = p.correct.astype(int)-p.base_correct.astype(int)
    p['outcome'] = np.select([p.net.eq(1),p.net.eq(-1),p.correct],['rescued','harmed','CC'],default='WW')
    mod = p[~p.arm.eq('A')].copy()
    for name,cols in [('class_effects',['arm','seed','class_id']),('acquisition_effects',['arm','seed','port','pass_group'])]:
        mod.groupby(cols).agg(n=('instance_id','size'),net=('net','sum'),
            rescued=('net',lambda x:int(x.eq(1).sum())),harmed=('net',lambda x:int(x.eq(-1).sum()))).reset_index().to_csv(HERE/(name+'.csv'),index=False)
    scopes = []
    for (arm,seed),part in mod.groupby(['arm','seed']):
        for label,pp in [('no_external',part[part.neighbors.eq(0)]),('external_available',part[part.neighbors.gt(0)])]:
            scopes.append(dict(arm=arm,seed=seed,scope=label,n=len(pp),net=int(pp.net.sum()),
                               **metrics(pp.class_id.to_numpy(),pp[['s0','s1','s2']].to_numpy())))
    pd.DataFrame(scopes).to_csv(HERE/'scope_effects.csv',index=False)
    paired,ci = [],[]
    comparisons = [(a,'A') for a in ARMS]+[('centered_own_full','uncentered_own_full'),('centered_own_full','centered_plus_bias')]
    for arm,reference in comparisons:
        differences = []
        for seed in SEEDS:
            aa = p[p.arm.eq(arm)&p.seed.eq(seed)].set_index('instance_id').loc[t.instance_id]
            bb = p[p.arm.eq(reference)&p.seed.eq(-1 if reference=='A' else seed)].set_index('instance_id').loc[t.instance_id]
            diff = aa.correct.to_numpy().astype(int)-bb.correct.to_numpy().astype(int)
            differences.append(diff)
            paired.append(dict(arm=arm,reference=reference,seed=seed,net=int(diff.sum()),
                OA_delta=float(diff.mean()*100),BA_delta=float(np.mean([diff[t.class_id.eq(c)].mean() for c in range(3)])*100)))
        averaged = np.mean(differences,axis=0)
        for block in ['pass_group','mmsi']:
            ci.append(dict(arm=arm,reference=reference,**intervals(t,averaged,block)))
    pd.DataFrame(paired).to_csv(HERE/'paired_metrics.csv',index=False)
    pd.DataFrame(ci).to_csv(HERE/'paired_intervals.csv',index=False)

    # Replay from saved input + model files against the persisted prediction CSV.
    replay = []
    for fold in range(6):
        with np.load(HERE/f'fold{fold}_inputs.npz') as f:
            d = {k:f[k] for k in f.files}
        test = d['test']
        np.testing.assert_array_equal(d['instance_ids'],t.instance_id)
        env = torch.tensor(d['env'][test])
        valid = torch.ones(len(test))
        for arm in ARMS:
            q = torch.tensor(d['qraw' if arm=='uncentered_own_full' else 'q'][test],dtype=torch.float32)
            for seed in SEEDS:
                model = (WithBias() if arm=='centered_plus_bias' else Anchored()).eval()
                model.load_state_dict(torch.load(HERE/'models'/f'fold{fold}_{arm}_{seed}.pt',map_location='cpu',weights_only=True))
                with torch.inference_mode():
                    score = d['base'][test]+model(env,q,valid).numpy()
                saved = p[p.fold.eq(fold)&p.arm.eq(arm)&p.seed.eq(seed)].set_index('instance_id').loc[t.iloc[test].instance_id]
                error = float(abs(score-saved[['s0','s1','s2']].to_numpy()).max())
                assert error < 2e-5
                np.testing.assert_array_equal(score.argmax(1),saved.prediction)
                np.testing.assert_array_equal(d['base'][test].argmax(1),saved.base_prediction)
                replay.append(dict(fold=fold,arm=arm,seed=seed,error=error,top1_changes=0))
    pd.DataFrame(replay).to_csv(HERE/'independent_replay.csv',index=False)

    anchor = mod[mod.arm.eq('centered_own_full')&mod.seed.eq(29)]
    choices = []
    for outcome in ['rescued','harmed','CC','WW']:
        selected = anchor[anchor.outcome.eq(outcome)&anchor.neighbors.eq(0)].sort_values('instance_id')
        no_external = True
        if selected.empty:
            selected = anchor[anchor.outcome.eq(outcome)].sort_values('instance_id')
            no_external = False
        if not selected.empty:
            row = selected.iloc[0]
            choices.append(dict(instance_id=row.instance_id,outcome=outcome,fold=int(row.fold),
                                selection='seed29, first ID per outcome; no-external preferred',no_external=no_external))
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    review,groups,panels = [],[],[]
    for case in choices:
        iid = case['instance_id']
        target = t.set_index('instance_id').loc[iid]
        index = int(np.flatnonzero(t.instance_id.eq(iid))[0])
        truth = int(target.class_id)
        im = [Image.open(EXP/'readout/native_targets'/f'{iid}_{pol}.png').convert('RGB') for pol in ['vv','vh']]
        canvas = Image.new('RGB',(1100,230+max(z.height for z in im)), 'white')
        ink = ImageDraw.Draw(canvas)
        ink.text((10,8),f'{iid} | {case["outcome"]} | truth={truth} {target.class_name} | {target.port}',font=font,fill='black')
        ink.text((10,34),f'fold={case["fold"]}, seed29; native pixels; no resize/enhancement; available neighbors={target.neighbors}',font=font,fill='black')
        rows = p[p.instance_id.eq(iid)&(p.seed.eq(29)|p.arm.eq('A'))].set_index('arm')
        for line,arm in enumerate(['A']+ARMS):
            rr = rows.loc[arm]
            scores = rr[['s0','s1','s2']].to_numpy(float)
            info = dict(**case,arm=arm,class_id=truth,prediction=int(rr.prediction),
                        margin=margin(scores,truth),port=target.port,pass_group=target.pass_group,mmsi=target.mmsi,
                        native_vv=str(EXP/'readout/native_targets'/f'{iid}_vv.png'),
                        native_vh=str(EXP/'readout/native_targets'/f'{iid}_vh.png'),
                        window_vv=target.target_window_vv,window_vh=target.target_window_vh)
            review.append(info)
            ink.text((10,60+line*25),f'{arm}: pred={int(rr.prediction)}, true margin={info["margin"]:+.5f}',font=font,fill='black')
        for x,pol,pixels in zip([10,560],['VV','VH'],im):
            ink.text((x,174),f'{pol} {pixels.width}x{pixels.height}',font=font,fill='black')
            canvas.paste(pixels,(x,200))
        path = HERE/f'case_{iid}.png'
        canvas.save(path)
        panels.append(str(path))
        with np.load(HERE/f'fold{case["fold"]}_inputs.npz') as f:
            env = torch.tensor(f['env'][index:index+1])
            q = f['q'][index]
            bp = int(f['base'][index].argmax())
        model = Anchored().eval()
        model.load_state_dict(torch.load(HERE/'models'/f'fold{case["fold"]}_centered_own_full_29.pt',map_location='cpu',weights_only=True))
        with torch.inference_mode():
            a = (.5*torch.tanh(model.net(env))).numpy()[0]
        rival = bp if bp!=truth else int(np.argmax(np.where(np.arange(3)==truth,-np.inf,rows.loc['A',['s0','s1','s2']].to_numpy(float))))
        for g in range(16,32):
            groups.append(dict(instance_id=iid,group=g,polarization='VV' if g<24 else 'VH',
                multiplier=float(1+a[g]),truth=truth,fixed_rival=rival,
                original_margin_contribution=float(q[g,truth]-q[g,rival]),
                delta_margin_contribution=float(a[g]*(q[g,truth]-q[g,rival]))))
    pd.DataFrame(review).to_csv(HERE/'review_cases.csv',index=False)
    pd.DataFrame(groups).to_csv(HERE/'review_group_contributions.csv',index=False)
    previous = pd.read_csv(STRONG/'baseline_predictions.csv').query('kind == "layer3"').set_index('instance_id').loc[t.instance_id]
    now = p[p.arm.eq('A')].set_index('instance_id').loc[t.instance_id]
    old_ok = previous.prediction.to_numpy()==t.class_id.to_numpy()
    now_ok = now.prediction.to_numpy()==t.class_id.to_numpy()
    summary = dict(review_cases=choices,panels=panels,independent_replayed_heads=len(replay),
        independent_replay_max=max(x['error'] for x in replay),
        old_partition_fixed_layer3=metrics(t.class_id.to_numpy(),previous[['s0','s1','s2']].to_numpy()),
        old_to_new_baseline_fixed=int((~old_ok&now_ok).sum()),old_to_new_baseline_lost=int((old_ok&~now_ok).sum()),
        caution='Same data, altered acquisition partition; no independent data confirmation. Cases are outcome-selected.')
    (HERE/'analysis_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(pd.DataFrame(paired).to_string(index=False),flush=True)
    print(pd.DataFrame(review).drop(columns=['native_vv','native_vh','window_vv','window_vh']).to_string(index=False),flush=True)
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    main()
