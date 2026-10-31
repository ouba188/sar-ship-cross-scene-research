"""Classification dependency, paired cases and native VV/VH inspection."""
import json
import html
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import experiment as joint

exp = joint.exp
HERE = joint.HERE
LABELS = ('bulk', 'fishing', 'general', 'chemical', 'container', 'crude', 'tug', 'offshore')


def restore(arm, seed):
    model = joint.make(arm, seed)
    status = model.load_state_dict(torch.load(HERE/'training'/f'{arm}_{seed}'/'m3.pt',
        map_location='cpu', weights_only=True), strict=False)
    assert not status.unexpected_keys
    return model


def margin(score, y):
    other = score.copy()
    other[np.arange(len(y)), y] = -np.inf
    return score[np.arange(len(y)), y]-other.max(1)


@torch.no_grad()
def dependency(rows, nf, rid, arrays, cache):
    ids = np.arange(nf, len(rows))
    y = np.array([rows[i]['label'] for i in ids])
    records = []
    for seed in joint.SEEDS:
        for arm in ('W_weighted', 'N_level_pattern'):
            model = restore(arm, seed)
            on = joint.predict(model, arm, cache, arrays, ids, rid, rows)
            saved = dict(np.load(HERE/'training'/f'{arm}_{seed}'/'final_development.npz'))
            error = float(np.max(abs(on['score']-saved['score'])))
            assert error < 1e-4, error
            for op in model.m3[0]: op.frequency = False
            off = joint.predict(model, arm, cache, arrays, ids, rid, rows)
            np.savez_compressed(HERE/'training'/f'{arm}_{seed}'/'frequency_off.npz', **off)
            ca, cb = on['score'].argmax(1)==y, off['score'].argmax(1)==y
            records.append(dict(arm=arm, seed=seed, replay_error=error,
                rescue=int((ca & ~cb).sum()), harm=int((~ca & cb).sum()),
                score_max_change=float(abs(on['score']-off['score']).max()),
                margin_mean_change=float((margin(on['score'],y)-margin(off['score'],y)).mean()),
                **{k: exp.metrics(y, on['score'])[k]-exp.metrics(y, off['score'])[k] for k in ('oa','ba','nll','brier')}))
            del model
            torch.cuda.empty_cache()
    pd.DataFrame(records).to_csv(HERE/'frequency_dependency.csv', index=False)
    print('DEPENDENCY', pd.DataFrame(records).to_string(index=False), flush=True)


@torch.no_grad()
def crossed_paths(rows, nf, rid, arrays, cache):
    """Separate comparison learning from value learning; no additional training."""
    arm, seed = 'N_level_pattern', 17
    checkpoint = torch.load(HERE/'training'/f'{arm}_{seed}'/'m3.pt', map_location='cpu', weights_only=True)
    train = np.array(json.loads((HERE/'split.json').read_text(encoding='utf-8'))['train'])
    records = []
    for comparison, values in ((False,False),(True,False),(False,True),(True,True)):
        model = joint.make(arm, seed)
        state = {k:v for k,v in checkpoint.items() if
                 (comparison and ('.selector.' in k or '.branch_bias' in k)) or
                 (values and ('.project.' in k or '.output.' in k))}
        model.load_state_dict(state, strict=False)
        name = f'comparison{int(comparison)}_value{int(values)}'
        for cohort, ids in (('source_train',train),('development',np.arange(nf,len(rows)))):
            z = joint.predict(model, arm, cache, arrays, ids, rid, rows)
            y = np.array([rows[i]['label'] for i in ids])
            records.append(dict(cohort=cohort, condition=name, **exp.metrics(y,z['score']),
                mean_update_ratio=float(z['update_ratio'].mean())))
            np.savez_compressed(HERE/f'cross_{cohort}_{name}.npz',**z)
        del model
        torch.cuda.empty_cache()
    pd.DataFrame(records).to_csv(HERE/'crossed_paths.csv',index=False)
    print('CROSSED_PATHS',pd.DataFrame(records).to_string(index=False),flush=True)


@torch.no_grad()
def trace_utilization(rows, nf, rid, arrays, cache):
    records = []
    ids = np.arange(nf,len(rows))
    for arm in ('W_weighted','N_level_pattern'):
        model = restore(arm,17)
        for first in range(0,len(ids),joint.BATCH):
            part = ids[first:first+joint.BATCH]
            _, on = joint.forward(model,arm,cache,arrays,part,rid,rows,details=True)
            for op in model.m3[0]: op.frequency=False
            _, off = joint.forward(model,arm,cache,arrays,part,rid,rows,details=True)
            for op in model.m3[0]: op.frequency=True
            for p in range(2):
                x = exp.tensor(arrays['x'],part)[:,p]
                delta = (on[p]['update']-off[p]['update']).flatten(1).norm(dim=1)
                relative = delta/x.flatten(1).norm(dim=1).clamp_min(1e-8)
                a,b = on[p]['reference_weights'],off[p]['reference_weights']
                live = a.sum(1)>0
                entropy = -(a.clamp_min(1e-20).log()*a).sum(1)
                l1 = (a-b).abs().sum(1)
                for j, index in enumerate(part):
                    active = live[j]
                    if not active.any(): continue
                    nrefs = on[p]['reference_valid'][j].sum().item()
                    same = torch.tensor([rows[k]['label']==rows[index]['label'] for k in rid],device=exp.DEVICE)
                    records.append(dict(arm=arm,id=rows[index]['id'],pol=('VV','VH')[p],
                        frequency_update_fraction=float(relative[j]),
                        reference_weight_l1=float(l1[j,active].mean()),
                        reference_entropy=float(entropy[j,active].mean()),
                        uniform_entropy=float(np.log(max(nrefs,1))),
                        same_class_reference_mass=float(a[j,same][:,active].sum(0).mean()),
                        same_class_reference_mass_off=float(b[j,same][:,active].sum(0).mean()),
                        frequency_branch_weight=float(torch.cat([
                            (weights[j,...,2:].sum(-1)*on[p]['transport'][r][j]).flatten()
                            for r,weights in enumerate(on[p]['branch_weights'])]).sum()/
                            torch.cat([v[j].flatten() for v in on[p]['transport']]).sum().clamp_min(1e-20))))
        del model
        torch.cuda.empty_cache()
    frame=pd.DataFrame(records)
    frame.to_csv(HERE/'utilization.csv',index=False)
    print('UTILIZATION',frame.groupby(['arm','pol']).mean(numeric_only=True).to_string(),flush=True)


def outcomes(seed):
    return {arm: dict(np.load(HERE/'training'/f'{arm}_{seed}'/'final_development.npz')) for arm in joint.ARMS}


@torch.no_grad()
def gallery(rows, nf, rid, arrays, cache, seed=17, only_frequency=False):
    data = outcomes(seed)
    ids = data['N_level_pattern']['ids']
    y = np.array([rows[i]['label'] for i in ids])
    ry = np.array([rows[i]['label'] for i in rid])
    off = dict(np.load(HERE/'training'/f'N_level_pattern_{seed}'/'frequency_off.npz'))
    initial = dict(np.load(HERE/'training'/f'N_level_pattern_{seed}'/'initial_development.npz'))
    new = data['N_level_pattern']['score']
    chosen, reasons = [], {}
    def choose(mask, effect, reason):
        positions = np.flatnonzero(mask)
        if len(positions):
            index = int(positions[np.argmax(abs(effect[positions]))])
            chosen.append(index)
            reasons.setdefault(index, []).append(reason)
    for base in ('S_spatial','W_weighted','L_level'):
        a, b = new.argmax(1)==y, data[base]['score'].argmax(1)==y
        change = margin(new,y)-margin(data[base]['score'],y)
        choose(a & ~b, change, f'classification rescue vs {base}')
        choose(~a & b, change, f'classification harm vs {base}')
    a,b = new.argmax(1)==y,initial['score'].argmax(1)==y
    change = margin(new,y)-margin(initial['score'],y)
    choose(a & ~b,change,'joint-training rescue vs own initial model')
    choose(~a & b,change,'joint-training harm vs own initial model')
    a, b = new.argmax(1)==y, off['score'].argmax(1)==y
    change = margin(new,y)-margin(off['score'],y)
    choose(a & ~b, change, 'frequency-on rescue with fixed trained weights')
    choose(~a & b, change, 'frequency-on harm with fixed trained weights')
    choose(a & b, change, 'still correct; largest frequency margin change')
    choose(~a & ~b, change, 'still wrong; largest frequency margin change')
    # Revisit prior representation-only examples even if classification never flips.
    for name in ('2042','12457','24730'):
        found = next((j for j,i in enumerate(ids) if str(rows[i]['id'])==name), None)
        if found is not None:
            chosen.append(found)
            reasons.setdefault(found, []).append('previous descriptor case')
    chosen = list(dict.fromkeys(chosen))
    if only_frequency:
        chosen = [j for j in chosen if (new[j].argmax()==y[j]) != (off['score'][j].argmax()==y[j])]
    model = restore('N_level_pattern',seed)
    notes = []
    gallery_dir = HERE/'gallery'
    gallery_dir.mkdir(exist_ok=True)
    for pos in chosen:
        idx = int(ids[pos]); record = rows[idx]
        score, details = joint.forward(model,'N_level_pattern',cache,arrays,np.array([idx]),rid,rows,details=True)
        for op in model.m3[0]: op.frequency = False
        score_off, details_off = joint.forward(model,'N_level_pattern',cache,arrays,np.array([idx]),rid,rows,details=True)
        for op in model.m3[0]: op.frequency = True
        valid = data['N_level_pattern']['valid'][pos]
        same = np.flatnonzero(valid & (ry==y[pos]))
        wrong = np.flatnonzero(valid & (ry!=y[pos]))
        same_id = int(same[np.argmax(data['N_level_pattern']['relation'][pos,same])]) if len(same) else None
        wrong_id = int(wrong[np.argmax(data['N_level_pattern']['relation'][pos,wrong])]) if len(wrong) else None
        indices = [idx, None if same_id is None else int(rid[same_id]), None if wrong_id is None else int(rid[wrong_id])]
        fig = plt.figure(figsize=(13,11))
        grid = GridSpec(3,4,figure=fig,height_ratios=(1,1,.85))
        for p in range(2):
            for col,index in enumerate(indices):
                axis = fig.add_subplot(grid[p,col])
                view = None if index is None else exp.data.prepare(rows[index])[p]
                if view is not None:
                    axis.imshow(view['pixels'],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                    masks = exp.tokens.slots(view['geometry'],view['footprint'])
                    axis.contour(masks[:8].sum(0),levels=[.05],colors=['#10ba85'],linewidths=.6)
                    axis.set_title(f'{("VV","VH")[p]} {("query","best same reference","best other reference")[col]}\n'
                        f'{rows[index]["id"]} / {LABELS[rows[index]["label"]]}',fontsize=9)
                else:
                    axis.text(.5,.5,'Unavailable polarization/reference',ha='center',fontsize=8)
                axis.set_xticks([]);axis.set_yticks([])
            axis = fig.add_subplot(grid[p,3])
            delta = (details[p]['update']-details_off[p]['update'])[0].norm(dim=0).cpu().numpy()
            image = axis.imshow(delta,cmap='magma',interpolation='nearest')
            fig.colorbar(image,ax=axis,fraction=.04)
            axis.set_title('Frequency-dependent writeback norm\n10x10 feature cells; not a physical map',fontsize=8)
            axis.set_xticks([]);axis.set_yticks([])
        axis = fig.add_subplot(grid[2,:2])
        for arm in joint.ARMS:
            axis.plot(range(8),data[arm]['score'][pos],marker='.',label=arm)
        axis.plot(range(8),off['score'][pos],linestyle='--',color='black',label='N frequency off')
        axis.plot(range(8),initial['score'][pos],linestyle=':',color='#97721d',label='N before training')
        axis.axvline(y[pos],color='gray',alpha=.3)
        axis.set_xticks(range(8),LABELS,rotation=30,fontsize=8)
        axis.set_title('Full classifier logits; shaded line = true class',fontsize=9)
        axis.legend(fontsize=6,ncol=2)
        axis = fig.add_subplot(grid[2,2:])
        rs = data['N_level_pattern']['relation'][pos].copy(); rs[~valid]=np.nan
        ro = off['relation'][pos].copy(); ro[~valid]=np.nan
        axis.plot(range(len(rid)),ro,marker='.',label='N frequency off')
        axis.plot(range(len(rid)),rs,marker='.',label='N frequency on')
        for k,c in enumerate(ry):
            if c==y[pos]: axis.axvspan(k-.3,k+.3,color='#159c68',alpha=.13)
        axis.set_xticks(range(len(rid)),[str(rows[i]['id']) for i in rid],rotation=65,fontsize=6)
        axis.set_title('Actual reference logits; green = same class',fontsize=9)
        axis.legend(fontsize=7)
        fig.suptitle(f'Query {record["id"]}, {LABELS[y[pos]]}: N predicts {LABELS[new[pos].argmax()]}\n'
            f'{"; ".join(reasons[pos])}\nNative-scale crop/pad; development, seed {seed}. Reference ranking is not classifier accuracy.',fontsize=10)
        fig.subplots_adjust(left=.045,right=.98,bottom=.1,top=.86,wspace=.3,hspace=.43)
        figure = gallery_dir/f'case_{record["id"]}_s{seed}.png'
        fig.savefig(figure,dpi=120,bbox_inches='tight');plt.close(fig)
        notes.append(dict(id=record['id'],seed=seed,row_index=idx,truth=int(y[pos]),port=record['port'],orbit=joint.probe.orbit(record),
            reasons=reasons[pos],paths={p:record[p]['image'] for p in ('vv','vh') if record.get(p)},
            predictions={arm:int(data[arm]['score'][pos].argmax()) for arm in joint.ARMS},
            initial_pred=int(initial['score'][pos].argmax()),
            initial_margin=float(margin(initial['score'][pos:pos+1],y[pos:pos+1])[0]),
            margins={arm:float(margin(data[arm]['score'][pos:pos+1],y[pos:pos+1])[0]) for arm in joint.ARMS},
            frequency_off_pred=int(off['score'][pos].argmax()),frequency_margin_delta=float(change[pos]),
            same_reference=None if same_id is None else rows[rid[same_id]]['id'],
            other_reference=None if wrong_id is None else rows[rid[wrong_id]]['id'],
            same_ref_logits=None if same_id is None else [float(off['relation'][pos,same_id]),float(data['N_level_pattern']['relation'][pos,same_id])],
            other_ref_logits=None if wrong_id is None else [float(off['relation'][pos,wrong_id]),float(data['N_level_pattern']['relation'][pos,wrong_id])],
            image=figure.name))
        print('CASE',record['id'],notes[-1]['predictions'],'frequency',change[pos],flush=True)
    exp.write(gallery_dir/f'cases_s{seed}.json',notes)
    notes = [n for file in sorted(gallery_dir.glob('cases_s*.json')) for n in json.loads(file.read_text(encoding='utf-8'))]
    exp.write(gallery_dir/'cases.json',notes)
    body = ''.join(f'<section><h2>{html.escape(str(n["id"]))}: {html.escape("; ".join(n["reasons"]))}</h2>'
        f'<img src="{n["image"]}" loading="lazy"><pre>{html.escape(json.dumps(n,ensure_ascii=False,indent=2))}</pre></section>' for n in notes)
    (gallery_dir/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>M3 joint readout cases</title>'
        '<style>body{font-family:system-ui;max-width:1350px;margin:24px auto;color:#242424}img{width:100%}'
        'section{border-top:1px solid #ccc;padding:24px 0}pre{white-space:pre-wrap}</style>'
        '<h1>M3 joint readout: full classification cases</h1><p><a href="../FINDINGS_ZH.md">完整中文实验报告</a> | '
        '<a href="case_24470_frequency.png">24470: 极化与频带开关对照</a></p><p>Native VV/VH, actual references, '
        'and fixed-model frequency ablation. Visual patterns are not named as physical mechanisms.</p>'+body,encoding='utf-8')


if __name__=='__main__':
    exp.current.setup(exp.DEVICE)
    rows,nf,refs,rid,arrays = exp.load()
    cache = joint.pair_cache()
    joint.tables(rows,rid)
    dependency(rows,nf,rid,arrays,cache)
    crossed_paths(rows,nf,rid,arrays,cache)
    trace_utilization(rows,nf,rid,arrays,cache)
    gallery(rows,nf,rid,arrays,cache)
    gallery(rows,nf,rid,arrays,cache,seed=43,only_frequency=True)
