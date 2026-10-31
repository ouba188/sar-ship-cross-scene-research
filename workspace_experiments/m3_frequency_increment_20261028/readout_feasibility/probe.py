"""CPU-only descriptor feasibility, fixed spatial matching and source-only scales."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run as exp
import analyze as previous
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sar_m123.m3 import sinkhorn

HERE = Path(__file__).resolve().parent
NAMES = ('cnn', 'freq_cos', 'freq_level', 'freq_shape_unit', 'freq_shape_damped',
         'freq_separate', 'space_cos', 'space_separate')
REGIONS = exp.tokens.REGIONS


def orbit(record):
    return '_'.join(record['pass_group'].split('_')[:2])


def unit(x):
    return x/np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-10)


def split(rows, nf, rid):
    ref_orbits = {orbit(rows[i]) for i in rid}
    candidates = sorted({orbit(r) for r in rows[:nf]}-ref_orbits)
    rng = np.random.default_rng(17)
    held = set(rng.permutation(candidates)[:max(1, len(candidates)//3)])
    probe = np.array([i for i in range(nf) if orbit(rows[i]) in held])
    held_vessels = {rows[i]['mmsi'] for i in probe}
    calibration = np.array([i for i in range(nf) if orbit(rows[i]) not in held and rows[i]['mmsi'] not in held_vessels])
    assert not {rows[i]['mmsi'] for i in calibration} & held_vessels
    return calibration, probe


def fit_scales(raw, occupied, calibration):
    occ = occupied[calibration, :, :, None, :]
    values = raw[calibration]
    meanbin = (values*occ).sum((0, 2))/np.maximum(occ.sum((0, 2)), 1)
    result = dict(meanbin=meanbin, level=np.zeros((2, 3, 3)), shape=np.zeros((2, 3, 3)))
    m = (values*occ).sum(-1)/np.maximum(occ.sum(-1), 1)
    rms = np.sqrt((((values-m[..., None])*occ)**2).sum(-1)/np.maximum(occ.sum(-1), 1))
    live = occupied[calibration].sum(-1) >= 4
    for p in range(2):
        for region, slots in enumerate(REGIONS):
            for b in range(3):
                a = m[:, p, slots, b][live[:, p, slots]]
                r = rms[:, p, slots, b][live[:, p, slots]]
                result['level'][p, region, b] = max(float(np.std(a)), 1e-6)
                result['shape'][p, region, b] = max(float(np.median(r)), 1e-6)
    return result


def compare(q, r, qo, ro, meanbin, level_scale, shape_scale):
    common = qo[:, None, :, None, :] & ro[None, :, None, :, :]
    n = common.sum(-1)
    qq, rr = q[:, None, :, None, :], r[None, :, None, :, :]
    mq = (qq*common).sum(-1)/np.maximum(n, 1)
    mr = (rr*common).sum(-1)/np.maximum(n, 1)
    cq, cr = (qq-mq[..., None])*common, (rr-mr[..., None])*common
    eq = (cq*cq).sum(-1)/np.maximum(n, 1)
    er = (cr*cr).sum(-1)/np.maximum(n, 1)
    cross = (cq*cr).sum(-1)/np.maximum(n, 1)
    shape = cross/np.maximum(np.sqrt(eq*er), 1e-12)
    damped = cross/np.sqrt((eq+shape_scale**2)*(er+shape_scale**2))
    level = np.exp(-.5*((mq-mr)/level_scale)**2)
    zq, zr = (qq-meanbin)*common, (rr-meanbin)*common
    cosine = (zq*zr).sum(-1)/np.maximum(np.sqrt((zq*zq).sum(-1)*(zr*zr).sum(-1)), 1e-12)
    result = dict(cos=(1+cosine)/2, level=level, shape_unit=(1+shape)/2,
                  shape_damped=(1+damped)/2, separate=.5*level+.25*(1+damped))
    return result, n >= 4


@torch.no_grad()
def evaluate(rows, ids, rid, arrays, scales):
    raw = np.asarray(arrays['layout'])
    occupied = np.asarray(arrays['occupied'])
    output, validity = [], []
    detail = []
    ry = np.array([rows[i]['label'] for i in rid])
    for first in range(0, len(ids), 20):
        batch = ids[first:first+20]
        legal = np.array([[rows[i]['mmsi'] != rows[j]['mmsi'] and orbit(rows[i]) != orbit(rows[j])
                           for j in rid] for i in batch])
        sums = np.zeros((len(batch), len(rid), len(NAMES)))
        counts = np.zeros((len(batch), len(rid)))
        for p in range(2):
            for region, slots in enumerate(REGIONS):
                deepq = np.asarray(arrays['deep'][batch])[:, p, slots]
                deepr = np.asarray(arrays['deep'][rid])[:, p, slots]
                fineq = np.asarray(arrays['fine'][batch])[:, p, slots, 0]
                finer = np.asarray(arrays['fine'][rid])[:, p, slots, 0]
                sim = (2*np.einsum('qid,kjd->qkij', deepq, deepr)+np.einsum('qid,kjd->qkij', fineq, finer))/3
                qm = np.asarray(arrays['mass'][batch])[:, p, slots]
                rm = np.asarray(arrays['mass'][rid])[:, p, slots]
                transport = sinkhorn(torch.from_numpy(sim), torch.from_numpy(qm[:, None]),
                                     torch.from_numpy(rm[None])).numpy()
                present = (qm[:, None, :, None]>0) & (rm[None, :, None, :]>0) & legal[:, :, None, None]
                qo, ro = occupied[batch][:, p, slots], occupied[rid][:, p, slots]
                for band in range(3):
                    frequency, good = compare(raw[batch][:, p, 0][:, slots, band],
                                              raw[rid][:, p, 0][:, slots, band], qo, ro,
                                              scales[0]['meanbin'][p, band], scales[0]['level'][p, region, band],
                                              scales[0]['shape'][p, region, band])
                    spatial, _ = compare(raw[batch][:, p, 1][:, slots, band],
                                         raw[rid][:, p, 1][:, slots, band], qo, ro,
                                         scales[1]['meanbin'][p, band], scales[1]['level'][p, region, band],
                                         scales[1]['shape'][p, region, band])
                    weight = transport*present*good
                    mass = weight.sum((-1, -2))
                    matrices = [(1+sim)/2]+[frequency[k] for k in ('cos', 'level', 'shape_unit', 'shape_damped', 'separate')]
                    matrices += [spatial['cos'], spatial['separate']]
                    part = np.stack([(weight*v).sum((-1, -2))/np.maximum(mass, 1e-20) for v in matrices], -1)
                    live = mass > 0
                    sums += part*live[..., None]
                    counts += live
                    for j, i in enumerate(batch):
                        same = (ry==rows[i]['label']) & live[j]
                        other = (ry!=rows[i]['label']) & live[j]
                        if same.any() and other.any():
                            for k, name in enumerate(NAMES):
                                detail.append(dict(id=rows[i]['id'], pol=('VV', 'VH')[p], region=('body', 'side', 'ends')[region],
                                    band=band, method=name, same_minus_other=float(part[j, same, k].max()-part[j, other, k].max())))
        output.append(sums/np.maximum(counts[..., None], 1))
        validity.append(counts > 0)
    return np.concatenate(output), np.concatenate(validity), detail


def rank_metrics(y, ry, score, legal):
    result = previous.relation_metrics(y, ry, score, legal)
    gaps, win, eligible = [], [], []
    for c, s, ok in zip(y, score, legal):
        positive, negative = ok & (ry==c), ok & (ry!=c)
        if not positive.any() or not negative.any():
            gaps.append(np.nan); win.append(False); eligible.append(False)
        else:
            gaps.append(float(s[positive].max()-s[negative].max()))
            win.append(bool(gaps[-1] > 0)); eligible.append(True)
    result['mean_best_pair_gap'] = float(np.nanmean(gaps))
    result['eligible'] = int(sum(eligible))
    # Highest-reference correctness includes only queries with a legal same-class reference.
    result['balanced_reference_top1'] = float(np.mean([np.mean(np.array(win)[(y==c)&eligible])
        for c in np.unique(y) if ((y==c)&eligible).any()]))
    return result, np.asarray(gaps), np.asarray(win), np.asarray(eligible)


def curves_and_tables(rows, cohorts, rid, outcomes):
    ry = np.array([rows[i]['label'] for i in rid])
    summary, cases, groups = [], [], []
    for cohort, ids in cohorts.items():
        scores, legal = outcomes[cohort]
        y = np.array([rows[i]['label'] for i in ids])
        methods = {name: scores[:, :, k] for k, name in enumerate(NAMES)}
        methods.update({f'cnn_plus_{name}': .75*scores[:, :, 0]+.25*scores[:, :, k]
                        for k, name in enumerate(NAMES) if k})
        old_gap = None
        for name, score in methods.items():
            m, gap, wins, eligible = rank_metrics(y, ry, score, legal)
            summary.append(dict(cohort=cohort, method=name, n=len(ids), **m))
            for j, i in enumerate(ids):
                candidates = np.flatnonzero(legal[j])
                top = int(candidates[np.argmax(score[j, candidates])]) if len(candidates) else -1
                cases.append(dict(cohort=cohort, id=rows[i]['id'], method=name, truth=rows[i]['label'],
                    port=rows[i]['port'], orbit=orbit(rows[i]), gap=float(gap[j]), eligible=bool(eligible[j]),
                    top_reference='' if top < 0 else rows[rid[top]]['id'],
                    reference_correct=bool(wins[j])))
            for field, values in (('class', y), ('orbit', np.array([orbit(rows[i]) for i in ids]))):
                for value in np.unique(values):
                    mask = values==value
                    gm = rank_metrics(y[mask], ry, score[mask], legal[mask])[0]
                    groups.append(dict(cohort=cohort, method=name, group=field, value=value, n=int(mask.sum()), **gm))
    pd.DataFrame(summary).to_csv(HERE/'metrics.csv', index=False)
    frame = pd.DataFrame(cases)
    frame.to_csv(HERE/'query_relations.csv', index=False)
    pd.DataFrame(groups).to_csv(HERE/'groups.csv', index=False)
    paired = []
    for cohort in cohorts:
        for base in ('cnn', 'cnn_plus_freq_cos', 'cnn_plus_freq_level'):
            a = frame[(frame.cohort==cohort)&(frame.method=='cnn_plus_freq_separate')].set_index('id')
            b = frame[(frame.cohort==cohort)&(frame.method==base)].set_index('id').loc[a.index]
            ca, cb = a.reference_correct, b.reference_correct
            paired.append(dict(cohort=cohort,base=base,new='cnn_plus_freq_separate',
                retrieval_rescue=int((ca & ~cb).sum()),retrieval_harm=int((~ca & cb).sum()),
                changed_reference=int((a.top_reference!=b.top_reference).sum())))
    pd.DataFrame(paired).to_csv(HERE/'paired_retrieval.csv',index=False)
    print(pd.DataFrame(paired).to_string(index=False),flush=True)
    print(pd.DataFrame(summary)[['cohort','method','n','eligible','map_at_r','reference_top1','balanced_reference_top1']].to_string(index=False), flush=True)


def saturation(raw, occupied, scales, ids):
    rows = []
    for p in range(2):
        for region, slots in enumerate(REGIONS):
            for b in range(3):
                value = raw[ids][:, p, slots, b]
                occ = occupied[ids][:, p, slots]
                n = occ.sum(-1)
                mu = (value*occ).sum(-1)/np.maximum(n, 1)
                rms = np.sqrt((((value-mu[..., None])*occ)**2).sum(-1)/np.maximum(n, 1))
                strength = rms/np.sqrt(rms*rms+scales['shape'][p, region, b]**2)
                rows.append(dict(pol=('VV','VH')[p], region=('body','side','ends')[region], band=b,
                    n_slots=int((n>=4).sum()), median_rms=float(np.median(rms[n>=4])),
                    source_rms_floor=float(scales['shape'][p, region, b]),
                    median_shape_strength=float(np.median(strength[n>=4]))))
    pd.DataFrame(rows).to_csv(HERE/'layout_variation.csv', index=False)


def gallery(rows, rid, arrays, scales, cohorts, outcomes):
    ids = cohorts['development']; scores, legal = outcomes['development']
    truth = np.array([rows[i]['label'] for i in ids]); ry = np.array([rows[i]['label'] for i in rid])
    old_scores = .75*scores[:,:,0]+.25*scores[:,:,1]
    new_scores = .75*scores[:,:,0]+.25*scores[:,:,5]
    old = rank_metrics(truth, ry, old_scores, legal)[1]
    new = rank_metrics(truth, ry, new_scores, legal)[1]
    delta = new-old
    chosen_ids = ['12457','2042','24730','3951','36077']
    chosen_ids += [rows[ids[int(np.nanargmax(delta))]]['id'], rows[ids[int(np.nanargmin(delta))]]['id']]
    for group in ((old<=0)&(new>0), (old>0)&(new<=0)):
        eligible=np.flatnonzero(group)
        if len(eligible):
            j=int(eligible[np.argmax(abs(delta[eligible]))])
            chosen_ids.append(rows[ids[j]]['id'])
    chosen_ids = list(dict.fromkeys(chosen_ids))
    notes = []
    layout = np.asarray(arrays['layout'])[:, :, 0]
    occupied = np.asarray(arrays['occupied'])
    for name in chosen_ids:
        pos = next(j for j,i in enumerate(ids) if rows[i]['id']==name)
        idx = int(ids[pos]); same = np.flatnonzero(legal[pos] & (ry==truth[pos])); other = np.flatnonzero(legal[pos] & (ry!=truth[pos]))
        # References selected once by old scores and retained for both readings.
        same = int(same[np.argmax(old_scores[pos,same])]); other = int(other[np.argmax(old_scores[pos,other])])
        indices = [idx,int(rid[same]),int(rid[other])]
        fig, axes = plt.subplots(6, 5, figsize=(13, 14))
        for p in range(2):
            views = [exp.data.prepare(rows[i])[p] for i in indices]
            allfields, masks = [], []
            for view in views:
                if view is None:
                    masks.append(None); allfields.append(None)
                    continue
                mm = exp.tokens.slots(view['geometry'], view['footprint'])
                masks.append(mm); allfields.append(exp.frequency.extract(view['raw'], mm, view['placement'])['energy_field'])
            # Choose a body slot using only valid area, not labels or improvements.
            slots = [int(np.asarray(arrays['mass'][i,p,:8]).argmax()) for i in indices]
            vmax = max(float(np.quantile(f[2][v['footprint']>0],.99)) for f,v in zip(allfields,views) if v is not None)
            for k,(index,view,field,slot) in enumerate(zip(indices,views,allfields,slots)):
                r=p*3+k
                if view is None:
                    for axis in axes[r]: axis.axis('off')
                    axes[r,0].text(.5,.5,f'{("VV","VH")[p]} {rows[index]["id"]}\nUnavailable polarization',ha='center')
                    continue
                axes[r,0].imshow(view['pixels'],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                axes[r,0].set_title(f'{("VV","VH")[p]} {("query","old best same","old best other")[k]} {rows[index]["id"]}',fontsize=8)
                axes[r,1].imshow(np.ma.masked_where(view['footprint']<=0,field[2]),cmap='magma',vmin=0,vmax=vmax,interpolation='nearest')
                axes[r,1].contour(masks[k][slot],levels=[.05],colors=['cyan'],linewidths=.6)
                axes[r,1].set_title(f'16px energy; body slot {slot}',fontsize=8)
                vals=layout[index,p,slot];occ=occupied[index,p,slot]
                mu=(vals*occ).sum(-1)/max(int(occ.sum()),1)
                centered=(vals-mu[:,None])*occ
                rms=np.sqrt((centered**2).sum(-1)/max(int(occ.sum()),1))
                tau=scales[0]['shape'][p,0]
                strength=rms/np.sqrt(rms*rms+tau*tau)
                axes[r,2].bar(range(3),mu,color='#287e96')
                axes[r,2].set(xticks=range(3),xticklabels=('4','8','16'),ylim=(0,12),title='Mean log energy')
                for b in range(3):axes[r,2].text(b,mu[b]+.15,f'{mu[b]:.2f}',ha='center',fontsize=7)
                shape=np.concatenate(list((centered/np.maximum(tau[:,None],1e-6)).reshape(3,4,4)),1)
                axes[r,3].imshow(shape,cmap='RdBu_r',vmin=-2,vmax=2,interpolation='nearest',aspect='equal')
                for border in (3.5,7.5):axes[r,3].axvline(border,color='black',linewidth=.7)
                axes[r,3].set_title('Centered 4x4 / source RMS\n4px | 8px | 16px; +/-2',fontsize=8)
                axes[r,4].bar(range(3),strength,color='#9b4766');axes[r,4].set(xticks=range(3),xticklabels=('4','8','16'),ylim=(0,1.1))
                axes[r,4].set_title('Pattern strength\nnot predictive confidence',fontsize=8)
                for b in range(3):axes[r,4].text(b,strength[b]+.02,f'{strength[b]:.2f}',ha='center',fontsize=7)
                for c in (0,1,3):axes[r,c].set_xticks([]);axes[r,c].set_yticks([])
        fig.suptitle(f'{name}: 75% CNN + 25% descriptor; ranking gap old={old[pos]:+.4f}, separate={new[pos]:+.4f}\n'
                     'Pictures retain original scale. Displayed max-area slots illustrate descriptors, not whole-system attribution.',fontsize=10)
        fig.tight_layout(rect=(0,0,1,.96));fig.savefig(HERE/f'case_{name}.png',dpi=115);plt.close(fig)
        notes.append(dict(id=name,truth=int(truth[pos]),old_gap=float(old[pos]),new_gap=float(new[pos]),
            same_reference=rows[rid[same]]['id'],other_reference=rows[rid[other]]['id'],
            old_same_score=float(old_scores[pos,same]),old_other_score=float(old_scores[pos,other]),
            new_same_score=float(new_scores[pos,same]),new_other_score=float(new_scores[pos,other]),
            fixed_pair_terms=dict(cnn=float(.75*(scores[pos,same,0]-scores[pos,other,0])),
                old_frequency=float(.25*(scores[pos,same,1]-scores[pos,other,1])),
                new_level=float(.125*(scores[pos,same,2]-scores[pos,other,2])),
                new_pattern=float(.125*(scores[pos,same,4]-scores[pos,other,4]))),
            selected_by='existing case' if name in ['12457','2042','24730','3951','36077'] else 'largest gap change',
            query_path={p:rows[idx][p]['image'] for p in ('vv','vh') if rows[idx].get(p)}))
        print('CASE',name,old[pos],new[pos],flush=True)
    exp.write(HERE/'cases.json',notes)


def main():
    rows,nf,refs,rid,arrays=exp.load()
    calibration,probe=split(rows,nf,rid)
    cohorts=dict(source_readout_holdout=probe,development=np.arange(nf,len(rows)))
    raw=np.asarray(arrays['layout']);occ=np.asarray(arrays['occupied'])
    scales=[fit_scales(raw[:,:,kind],occ,calibration) for kind in range(2)]
    exp.write(HERE/'split.json',dict(calibration_ids=[rows[i]['id'] for i in calibration],
        source_probe_ids=[rows[i]['id'] for i in probe],calibration_n=len(calibration),probe_n=len(probe),
        probe_orbits=sorted({orbit(rows[i]) for i in probe}),
        calibration_orbits=sorted({orbit(rows[i]) for i in calibration}),
        reference_ids=[rows[i]['id'] for i in rid],
        source_probe_classes={str(c):sum(rows[i]['label']==c for i in probe) for c in range(8)}))
    exp.write(HERE/'scales.json',[{k:v.tolist() for k,v in s.items()} for s in scales])
    outcomes={}
    details=[]
    for name,ids in cohorts.items():
        if '--gallery-only' in sys.argv:
            saved=np.load(HERE/f'{name}.npz')
            score,valid=saved['score'],saved['valid']
            detail=[]
        else:
            score,valid,detail=evaluate(rows,ids,rid,arrays,scales)
        outcomes[name]=(score,valid)
        np.savez_compressed(HERE/f'{name}.npz',score=score,valid=valid,indices=ids)
        details += [dict(cohort=name,**r) for r in detail]
        print('EVALUATED',name,len(ids),flush=True)
    if details:
        pd.DataFrame(details).to_csv(HERE/'polar_region_band.csv',index=False)
    curves_and_tables(rows,cohorts,rid,outcomes)
    saturation(raw[:,:,0],occ,scales[0],cohorts['development'])
    gallery(rows,rid,arrays,scales,cohorts,outcomes)


if __name__=='__main__':
    torch.set_num_threads(2)
    main()
