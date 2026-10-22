"""Read saved readers/residuals, replay 54 heads, and trace the changed error set."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import sys
sys.dont_write_bytecode = True
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from osgeo import gdal
from run_strong import HERE, EXP, KINDS, ARMS, SEEDS, Relation, inputs_for, metrics


def score_margin(row, prefix):
    values = np.array([row[f'{prefix}{c}'] for c in range(3)], dtype=float)
    truth = int(row['class_id'])
    return float(values[truth] - np.delete(values, truth).max())


def render_cases(targets, edges, cases):
    """Reuse native target PNGs and only read recorded neighbor TIFF windows."""
    gdal.UseExceptions()
    gdal.SetConfigOption('GDAL_PAM_ENABLED', 'NO')
    gdal.SetCacheMax(64 * 1024**2)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 15)
    panels, provenance = [], []
    for number, case in enumerate(cases, 1):
        iid = case['instance_id']
        row = targets.loc[iid]
        es = edges[edges.instance_id.eq(iid)].sort_values('slot').head(3)
        pairs = []
        ims = [Image.open(EXP/'readout/native_targets'/f'{iid}_{pol}.png').convert('RGB') for pol in ('vv', 'vh')]
        pairs.append(('target', ims))
        for edge in es.itertuples():
            ims = []
            for pol in ('vv', 'vh'):
                path = getattr(edge, 'raster_' + pol)
                window = json.loads(getattr(edge, 'read_window_' + pol))
                ds = gdal.Open(path)
                pixels = ds.ReadAsArray(*window)
                ds = None
                im = Image.fromarray(pixels).convert('RGB')
                corners = np.array(json.loads(edge.source_points)) - np.array(window[:2])
                ImageDraw.Draw(im).line([tuple(x) for x in np.vstack([corners, corners[0]])], fill='#ffb020', width=1)
                ims.append(im)
            pairs.append((f'{edge.detection_id} {edge.distance_m:.0f}m {edge.land_action}', ims))
            provenance.append(dict(instance_id=iid,slot=edge.slot,detection_id=edge.detection_id,
                                   raster_vv=edge.raster_vv,raster_vh=edge.raster_vh,
                                   read_window_vv=edge.read_window_vv,read_window_vh=edge.read_window_vh,
                                   distance_m=edge.distance_m,land_action=edge.land_action))
        widths = [max(300, max(im.width for im in pair)+16, int(font.getlength(name))+16) for name, pair in pairs]
        height = max(im.height for _, pair in pairs for im in pair)
        canvas = Image.new('RGB', (max(1200, sum(widths)), 230 + 2*(height+34)), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((8,8), f'{iid} | {row.port} | truth={row.class_name} | N={row.neighbors} | {case["selection_reason"]}',font=font,fill='black')
        draw.text((8,31), f'seed29: old A {case["old_base_prediction"]}; old split {case["old_split_prediction"]}; strong A {case["base_prediction"]}; strong split {case["prediction"]}',font=font,fill='black')
        draw.text((8,54), f'true margin: old A {case["old_base_margin"]:+.3f}; old split {case["old_split_margin"]:+.3f}; strong A {case["base_margin"]:+.3f}; strong split {case["margin"]:+.3f}',font=font,fill='black')
        draw.text((8,77), f'whole_unit {case["whole_prediction"]}; split_role {case["role_prediction"]}; 0=bulk, 1=fishing, 2=general',font=font,fill='black')
        draw.text((8,100), 'Native pixels; VV/VH separate. Nearest <=3 actual neighbors shown, not selected by effect. No neighbor labels.',font=font,fill='black')
        x = 0
        for width, (name, pair) in zip(widths, pairs):
            draw.text((x+8,130), name,font=font,fill='black')
            for index, (pol, im) in enumerate(zip(('VV','VH'),pair)):
                y = 160 + index*(height+34)
                draw.text((x+8,y),pol,font=font,fill='black')
                canvas.paste(im,(x+8,y+24))
            x += width
        destination = HERE/f'outcome_review_{number}.png'
        canvas.save(destination)
        panels.append(str(destination))
    pd.DataFrame(provenance).to_csv(HERE/'review_neighbors.csv',index=False)
    return panels


def main():
    started = time.perf_counter()
    torch.set_num_threads(2)
    t = pd.read_csv(EXP/'readout/targets.csv',dtype={'mmsi':str})
    p = pd.read_csv(HERE/'predictions.csv')
    old = pd.read_csv(EXP/'predictions.csv').query('arm == "split_unit"')
    readers = pd.read_csv(HERE/'baseline_predictions.csv')
    choices = pd.read_csv(HERE/'baseline_candidates.csv')
    selected = pd.read_csv(HERE/'selections.csv').set_index('fold')
    identities = t.mmsi.to_numpy()
    bins = t.pass_group.map(pd.read_csv(EXP/'pass_assignment.csv').set_index('pass_group').group_bin).to_numpy()
    metadata = t[['instance_id','mmsi','port','pass_group','class_name']]
    p = p.merge(metadata,on='instance_id',validate='many_to_one')
    p['correct'] = p.prediction.eq(p.class_id)
    p['base_correct'] = p.base_prediction.eq(p.class_id)
    p['net'] = p.correct.astype(int)-p.base_correct.astype(int)
    p['kind'] = np.select([p.net.eq(1),p.net.eq(-1),p.correct],['rescued','harmed','CC'],default='WW')
    reader_rows = []
    for kind, q in readers.groupby('kind'):
        assert len(q)==len(t) and q.instance_id.is_unique
        reader_rows.append(dict(kind=kind,**metrics(q.class_id.to_numpy(),q[['s0','s1','s2']].to_numpy())))
    anchor = p.query('arm == "split_unit" and seed == 29').set_index('instance_id').loc[t.instance_id].reset_index()
    reader_rows.append(dict(kind='source_selected',**metrics(anchor.class_id.to_numpy(),anchor[['b0','b1','b2']].to_numpy())))
    pd.DataFrame(reader_rows).to_csv(HERE/'reader_pooled.csv',index=False)

    selection_rows, replays = [], []
    for fold in range(6):
        c = choices[choices.fold.eq(fold)].sort_values('val_nll',kind='stable')
        s = selected.loc[fold]
        assert c.iloc[0]['kind']==s['kind'] and float(c.iloc[0].alpha)==float(s.alpha)
        selection_rows.append(dict(fold=fold,selected=s['kind'],alpha=s.alpha,val_nll=s.val_nll,
                                   runner_up=c.iloc[1]['kind'],validation_gap=float(c.iloc[1].val_nll-s.val_nll)))
        with np.load(EXP/f'fold{fold}_inputs.npz') as data:
            d = dict(data)
        with np.load(HERE/f'fold{fold}_base.npz') as data:
            strong = dict(data)
        for role in ['train','validation','test']:
            assert np.array_equal(d[role],strong[role])
        assert str(strong['selected_kind'])==s['kind'] and float(strong['selected_alpha'])==float(s.alpha)
        tr, va, te = [d[k] for k in ['train','validation','test']]
        for aa,bb in [(tr,va),(tr,te),(va,te)]:
            assert not set(identities[aa])&set(identities[bb])
            assert not set(bins[aa])&set(bins[bb])
        covered=[]
        for b in sorted(set(bins[tr])):
            held=tr[bins[tr]==b]
            fit=tr[(bins[tr]!=b)&~np.isin(identities[tr],identities[held])]
            assert not set(identities[fit])&set(identities[held])
            assert set(t.iloc[fit].class_id)=={0,1,2}
            covered.extend(held)
        assert sorted(covered)==sorted(tr.tolist())
        assert np.isfinite(strong['oof'][tr]).all() and np.all(np.any(strong['oof'][tr]!=0,axis=1))
        for arm in ARMS:
            inputs=inputs_for(arm,d,'cpu')
            for seed in SEEDS:
                model=Relation('split_unit' if arm=='split_role' else arm).eval()
                model.load_state_dict(torch.load(HERE/'models'/f'fold{fold}_{arm}_{seed}.pt',map_location='cpu',weights_only=True))
                with torch.inference_mode():
                    delta=model(*inputs).numpy()
                assert not delta[~d['m'].any(1)].any()
                score=strong['base'][te]+delta[te]
                q=p[p.fold.eq(fold)&p.arm.eq(arm)&p.seed.eq(seed)].set_index('instance_id').loc[t.iloc[te].instance_id]
                error=float(np.max(np.abs(score-q[['s0','s1','s2']].to_numpy())))
                assert error<2e-5 and np.array_equal(score.argmax(1),q.prediction.to_numpy()),(fold,arm,seed,error)
                replays.append(dict(fold=fold,arm=arm,seed=seed,max_abs_error=error,top1_same=True))
    pd.DataFrame(selection_rows).to_csv(HERE/'selection_audit.csv',index=False)
    pd.DataFrame(replays).to_csv(HERE/'replay.csv',index=False)

    p.groupby(['arm','seed','class_id']).agg(n=('instance_id','size'),net=('net','sum'),
         rescued=('net',lambda a:int(a.eq(1).sum())),harmed=('net',lambda a:int(a.eq(-1).sum()))).reset_index().to_csv(HERE/'class_effects.csv',index=False)
    p.groupby(['arm','seed','port','pass_group']).agg(n=('instance_id','size'),net=('net','sum')).reset_index().to_csv(HERE/'acquisition_effects.csv',index=False)
    old_columns=old[['instance_id','seed','class_id','prediction','base_prediction','s0','s1','s2','b0','b1','b2']].rename(
        columns={k:'old_'+k for k in ['class_id','prediction','base_prediction','s0','s1','s2','b0','b1','b2']})
    cross=p.merge(old_columns,on=['instance_id','seed'],validate='many_to_one')
    assert (cross.class_id==cross.old_class_id).all()
    cross['old_A_correct']=cross.old_base_prediction.eq(cross.class_id)
    cross['old_split_correct']=cross.old_prediction.eq(cross.class_id)
    cross['old_split_rescue']=~cross.old_A_correct&cross.old_split_correct
    cross['resolved_by_strong_A']=cross.old_split_rescue&cross.base_correct
    cross['remaining_old_rescue']=cross.old_split_rescue&~cross.base_correct
    cross['remaining_recovered']=cross.remaining_old_rescue&cross.correct
    cross.to_csv(HERE/'error_transition_rows.csv',index=False)
    transitions=[]
    for (arm,seed),q in cross.groupby(['arm','seed']):
        transitions.append(dict(arm=arm,seed=seed,old_A_errors=int((~q.old_A_correct).sum()),
            strong_A_errors=int((~q.base_correct).sum()),
            old_A_error_fixed=int((~q.old_A_correct&q.base_correct).sum()),
            old_A_correct_lost=int((q.old_A_correct&~q.base_correct).sum()),
            old_split_rescues=int(q.old_split_rescue.sum()),resolved=int(q.resolved_by_strong_A.sum()),
            remaining=int(q.remaining_old_rescue.sum()),remaining_recovered=int(q.remaining_recovered.sum()),
            resolved_reharmed=int((q.resolved_by_strong_A&~q.correct).sum()),
            rescued=int(q.net.eq(1).sum()),harmed=int(q.net.eq(-1).sum()),net=int(q.net.sum())))
    pd.DataFrame(transitions).to_csv(HERE/'error_transition_summary.csv',index=False)
    a=cross.query('arm == "split_unit" and seed == 29').copy()
    a.groupby('class_id').agg(n=('instance_id','size'),old_A_errors=('old_A_correct',lambda x:int((~x).sum())),
        strong_A_errors=('base_correct',lambda x:int((~x).sum())),old_split_rescues=('old_split_rescue','sum'),
        resolved=('resolved_by_strong_A','sum'),remaining=('remaining_old_rescue','sum')).reset_index().to_csv(HERE/'baseline_gap_by_class.csv',index=False)
    scopes=[]
    for name,q in [('all',a),('context',a[a.neighbors.gt(0)]),('no_context',a[a.neighbors.eq(0)])]:
        scopes.append(dict(scope=name,n=len(q),old_A_errors=int((~q.old_A_correct).sum()),
                           strong_A_errors=int((~q.base_correct).sum()),
                           fixed=int((~q.old_A_correct&q.base_correct).sum()),lost=int((q.old_A_correct&~q.base_correct).sum())))
    pd.DataFrame(scopes).to_csv(HERE/'baseline_gap_by_scope.csv',index=False)

    chosen=[]
    eligible=a[a.neighbors.gt(0)]
    for kind in ['rescued','harmed','CC','WW']:
        found=eligible[eligible.kind.eq(kind)].sort_values('instance_id').head(1)
        for r in found.to_dict('records'):
            r['selection_reason']=f'strong split {kind}, first ID at seed29';chosen.append(r)
    for flag in ['resolved_by_strong_A','remaining_old_rescue']:
        found=eligible[eligible[flag]&~eligible.instance_id.isin([r['instance_id'] for r in chosen])].sort_values('instance_id').head(1)
        for r in found.to_dict('records'):
            r['selection_reason']=f'old split rescue: {flag}, first remaining ID';chosen.append(r)
    cases=[]
    for r in chosen:
        r['old_base_margin']=score_margin(r,'old_b')
        r['old_split_margin']=score_margin(r,'old_s')
        r['base_margin']=score_margin(r,'b')
        r['margin']=score_margin(r,'s')
        r['old_split_prediction']=r['old_prediction']
        for arm,key in [('whole_unit','whole_prediction'),('split_role','role_prediction')]:
            r[key]=int(p[p.instance_id.eq(r['instance_id'])&p.seed.eq(29)&p.arm.eq(arm)].iloc[0].prediction)
        cases.append(r)
    pd.DataFrame(cases).to_csv(HERE/'review_selection.csv',index=False)
    p[p.instance_id.isin([r['instance_id'] for r in cases])].to_csv(HERE/'review_cases.csv',index=False)
    panels=render_cases(t.set_index('instance_id'),pd.read_csv(EXP/'readout/neighbors.csv'),cases)
    summary=dict(heads_replayed=len(replays),replay_device='cpu',replay_atol=2e-5,
        replay_max_abs_error=max(r['max_abs_error'] for r in replays),top1_all_same=True,
        selection_matches_source_validation=True,inner_split_membership_valid=True,
        context_inputs='unchanged expanded fold inputs; all three bag arms ignore query',
        limitation='saved selected-A scores replayed, no ridge/PCA refit; not an end-to-end OOF of context representation',
        reviewed_ids=[r['instance_id'] for r in cases],panels=panels,seconds=time.perf_counter()-started)
    (HERE/'analysis_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(pd.DataFrame(reader_rows).to_string(index=False),flush=True)
    print(pd.DataFrame(transitions).to_string(index=False),flush=True)
    print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
