"""Paired effects, fixed-model context substitutions, and original-image review."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from osgeo import gdal
from run_anchored import HERE, EXP, STRONG, ARMS, Anchored, SEEDS, metrics
sys.path.insert(0, str(EXP))
from analyze_expanded import clustered_ci


def render(t, edges, p, ids, group_table, arms=ARMS, prefix='case'):
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 15)
    indexed = t.set_index('instance_id')
    datasets = {}
    for iid in ids:
        r = indexed.loc[iid]
        canvas = Image.new('RGB', (1350, 620), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((8,5), f'{iid} {r.port} truth={r.class_name} N={r.neighbors} seed29', font=font, fill='black')
        scores = p[p.instance_id.eq(iid)&p.seed.eq(29)].set_index('arm')
        for k, arm in enumerate(arms):
            a = scores.loc[arm]
            s = a[['s0','s1','s2']].to_numpy(float)
            c = int(a.class_id)
            margin = s[c]-np.delete(s,c).max()
            draw.text((8,30+21*k), f'{arm}: {int(a.base_prediction)} -> {int(a.prediction)}, margin={margin:+.4f}', font=font, fill='black')
        g = group_table[group_table.instance_id.eq(iid)&group_table.seed.eq(29)].copy()
        g['strength'] = g.margin_increment.abs()
        for k, a in enumerate(g.sort_values('strength',ascending=False).head(4).itertuples()):
            draw.text((8,105+20*k), f'group{a.group}: a={a.modulation:+.3f}, target-margin={a.target_margin:+.3f}, change={a.margin_increment:+.4f}', font=font, fill='black')
        for k, pol in enumerate(['vv','vh']):
            im = Image.open(EXP/'readout/native_targets'/f'{iid}_{pol}.png').convert('RGB')
            draw.text((8+k*175,208), 'target '+pol.upper(), font=font, fill='black')
            canvas.paste(im,(8+k*175,235))
        shown = edges[edges.instance_id.eq(iid)].sort_values('slot').head(3)
        for j, e in enumerate(shown.itertuples()):
            x = 410+j*310
            draw.text((x,110),f'{e.detection_id} {e.distance_m:.0f}m',font=font,fill='black')
            draw.text((x,132),e.land_action,font=font,fill='black')
            for k,pol in enumerate(['vv','vh']):
                path = getattr(e,'raster_'+pol)
                if path not in datasets:
                    datasets[path] = gdal.Open(path)
                win = json.loads(getattr(e,'read_window_'+pol))
                im = Image.fromarray(datasets[path].ReadAsArray(*win)).convert('RGB')
                pts = np.asarray(json.loads(e.source_points))-np.asarray(win[:2])
                ImageDraw.Draw(im).line([tuple(v) for v in np.vstack([pts,pts[0]])],fill='#ffb020',width=1)
                draw.text((x,165+k*185),pol.upper(),font=font,fill='black')
                canvas.paste(im,(x,190+k*185))
        draw.text((8,570),'Native pixels, no model resizing. 0=bulk,1=fishing,2=general. Nearest three neighbors only; model uses all valid nodes.',font=font,fill='black')
        draw.text((8,592),'Group contributions are algebraic model evidence, not identified physical scattering mechanisms.',font=font,fill='black')
        canvas.save(HERE/f'{prefix}_{iid}.png')


def main():
    torch.set_num_threads(2)
    gdal.UseExceptions()
    gdal.SetConfigOption('GDAL_PAM_ENABLED','NO')
    gdal.SetCacheMax(64*1024**2)
    t = pd.read_csv(EXP/'readout/targets.csv',dtype={'mmsi':str})
    p = pd.read_csv(HERE/'predictions.csv')
    edges = pd.read_csv(EXP/'readout/neighbors.csv')
    p = p.merge(t[['instance_id','mmsi','port','pass_group']],on='instance_id',validate='many_to_one')
    p['net'] = (p.prediction == p.class_id).astype(int)-(p.base_prediction == p.class_id).astype(int)
    p['kind'] = np.select([p.net.eq(1),p.net.eq(-1),p.prediction.eq(p.class_id)],['rescued','harmed','correct'],default='wrong')
    p.groupby(['arm','seed','class_id']).agg(n=('instance_id','size'),net=('net','sum')).reset_index().to_csv(HERE/'class_effects.csv',index=False)
    p.groupby(['arm','seed','pass_group','port']).agg(n=('instance_id','size'),net=('net','sum')).reset_index().to_csv(HERE/'acquisition_effects.csv',index=False)
    intervals = []
    other = pd.read_csv(STRONG/'predictions.csv')
    comparisons = {'base':None,'constant_env':p[p.arm.eq('constant_env')],
        'scene_replacement':p[p.arm.eq('scene_replacement')],'additive_split':other[other.arm.eq('split_unit')]}
    aa = p[p.arm.eq('anchored')].set_index(['seed','instance_id'])
    for name, ref in comparisons.items():
        if ref is None:
            diff = aa.net
        else:
            b = ref.set_index(['seed','instance_id']).reindex(aa.index)
            diff = aa.prediction.eq(aa.class_id).astype(int)-b.prediction.eq(b.class_id).astype(int)
        avg = diff.groupby('instance_id').mean().reindex(t.instance_id).to_numpy()
        for block in ['pass_group','mmsi']:
            intervals.append(dict(reference=name,block=block,**clustered_ci(t,avg,block)))
    pd.DataFrame(intervals).to_csv(HERE/'paired_intervals.csv',index=False)
    pick = p[p.arm.eq('anchored')&p.seed.eq(29)&p.neighbors.gt(0)]
    chosen = []
    for kind in ['rescued','harmed','correct','wrong']:
        chosen.extend(pick[pick.kind.eq(kind)].sort_values('instance_id').head(2).instance_id)
    p[p.instance_id.isin(chosen)].to_csv(HERE/'review_cases.csv',index=False)
    replay, fixed, groups, gain_summary = [], [], [], []
    y = t.class_id.to_numpy()
    for fold in range(6):
        d = dict(np.load(HERE/f'fold{fold}_inputs.npz'))
        test = d['test']
        q = torch.tensor(d['q'][test],dtype=torch.float32,device='cuda')
        valid = torch.tensor(d['has'][test],dtype=torch.float32,device='cuda')
        for seed in SEEDS:
            for arm in ARMS:
                model = Anchored().cuda().eval()
                model.load_state_dict(torch.load(HERE/'models'/f'fold{fold}_{arm}_{seed}.pt',weights_only=True,map_location='cpu'))
                env = torch.tensor(d['env' if arm == 'anchored' else arm][test],device='cuda',dtype=torch.float32)
                with torch.no_grad():
                    score = d['base'][test]+model(env,q,valid).cpu().numpy()
                ref = p[p.fold.eq(fold)&p.arm.eq(arm)&p.seed.eq(seed)].set_index('instance_id').loc[t.iloc[test].instance_id]
                err = float(np.abs(score-ref[['s0','s1','s2']].to_numpy()).max())
                assert err < 1e-4 and np.array_equal(score.argmax(1),ref.prediction)
                replay.append(dict(fold=fold,arm=arm,seed=seed,error=err))
                if arm != 'anchored':
                    continue
                with torch.no_grad():
                    modulation = (.5*torch.tanh(model.net(env))).cpu().numpy()
                active = d['has'][test]
                for group in range(32):
                    gain_summary.append(dict(fold=fold,seed=seed,group=group,n=int(active.sum()),
                        mean=float(modulation[active,group].mean()),sd=float(modulation[active,group].std()),
                        target_evidence_rms=float(np.sqrt((d['q'][test][active,group]**2).mean()))))
                for k,i in enumerate(test):
                    if t.iloc[i].instance_id in chosen:
                        rival = int(np.argmax(np.where(np.arange(3)==y[i],-np.inf,d['base'][i])))
                        for group in range(32):
                            evidence = d['q'][i,group,y[i]]-d['q'][i,group,rival]
                            groups.append(dict(instance_id=t.iloc[i].instance_id,fold=fold,seed=seed,group=group,
                                true_class=int(y[i]),base_rival=rival,target_margin=evidence,modulation=float(modulation[k,group]),
                                margin_increment=float(modulation[k,group]*evidence)))
                for variant in ['constant_env','scene_replacement']:
                    replacement = torch.tensor(d[variant][test],dtype=torch.float32,device='cuda')
                    with torch.no_grad():
                        ss = d['base'][test]+model(replacement,q,valid).cpu().numpy()
                    for k,i in enumerate(test):
                        fixed.append(dict(fold=fold,seed=seed,arm=variant,instance_id=t.iloc[i].instance_id,class_id=int(y[i]),
                            prediction=int(ss[k].argmax()),base_prediction=int(d['base'][i].argmax()),
                            **{f's{c}':float(ss[k,c]) for c in range(3)}))
    pd.DataFrame(replay).to_csv(HERE/'replay.csv',index=False)
    fg = pd.DataFrame(fixed)
    fg.to_csv(HERE/'fixed_context_predictions.csv',index=False)
    fm=[]
    for (arm,seed),part in fg.groupby(['arm','seed']):
        fm.append(dict(arm=arm,seed=seed,**metrics(part.class_id.to_numpy(),part[['s0','s1','s2']].to_numpy()),
            net=int((part.prediction==part.class_id).sum()-(part.base_prediction==part.class_id).sum())))
    pd.DataFrame(fm).to_csv(HERE/'fixed_context_metrics.csv',index=False)
    gt = pd.DataFrame(groups)
    gt.to_csv(HERE/'case_group_contributions.csv',index=False)
    pd.DataFrame(gain_summary).to_csv(HERE/'modulation_summary.csv',index=False)
    render(t,edges,p,chosen,gt)
    summary = dict(heads_replayed=len(replay),max_error=max(x['error'] for x in replay),review_ids=chosen,
        mechanism_limit='Zero-centered-target implies zero context delta, not literal absence of a ship; feature groups are not named physical factors.',
        ci_note='Separate pass/MMSI paired block bootstrap, not joint; current reused development cohort.')
    (HERE/'analysis_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(pd.DataFrame(fm).to_string(index=False),flush=True)
    print(pd.DataFrame(intervals).to_string(index=False),flush=True)


if __name__ == '__main__':
    main()
