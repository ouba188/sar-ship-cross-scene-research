"""Read frozen source predictions/flux; never fits or extracts features."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import run_readout as run
from extract_features import native_image


def score(g, prefix='s'):
    return g[[f'{prefix}{c}' for c in range(8)]].to_numpy(float)


def compare(a, b, reference):
    a = a.set_index('instance_id').sort_index()
    b = b.set_index('instance_id').sort_index()
    assert a.index.equals(b.index)
    np.testing.assert_array_equal(a.class_id, b.class_id)
    np.testing.assert_array_equal(score(a, 'b'), score(b, 'b'))
    before = score(a, 'b') if reference == 'B' else score(b)
    after = score(a)
    y = a.class_id.to_numpy()
    pre, post = before.argmax(1), after.argmax(1)
    old, new = run.P.assessment(y, before), run.P.assessment(y, after)
    ac, bc = post == y, pre == y
    metrics = dict(**new, rescued=int((ac & ~bc).sum()), harmed=int((~ac & bc).sum()),
        net=int(ac.sum()-bc.sum()), top1_changes=int((post != pre).sum()),
        **{f'delta_{k}': new[k]-old[k] for k in ['OA','BA','NLL','balanced_NLL','Brier']})
    changes = a.reset_index().copy()
    changes['reference'] = reference
    changes['reference_prediction'] = pre
    changes['outcome'] = np.select([ac & ~bc, ~ac & bc, ac & bc], ['rescued','harmed','CC'], default='WW')
    changes['net'] = ac.astype(int)-bc.astype(int)
    return metrics, changes


def fluxes(predictions):
    matrices, checks = {}, []
    paths = pd.read_csv(ROOT / 'validation_paths.csv').set_index(['seed','arm','instance_id'])
    for seed in run.SEEDS:
        for arm in run.ARMS:
            with np.load(ROOT/'validation_flux'/f'{arm}_{seed}.npz') as z:
                ids, pairs, h, b, s = z['sample_ids'], z['pairs'], z['h'], z['B'], z['s']
                epoch = int(z['epoch'])
            g = predictions[predictions.seed.eq(seed) & predictions.arm.eq(arm) & predictions.role.eq('validation')].set_index('instance_id').loc[ids]
            np.testing.assert_allclose(s, score(g), atol=1e-12, rtol=0)
            np.testing.assert_allclose(b, score(g, 'b'), atol=1e-12, rtol=0)
            hm = np.zeros((len(ids),8,8))
            hm[:,pairs[:,0],pairs[:,1]] = h
            hm[:,pairs[:,1],pairs[:,0]] = -h
            err = float(np.max(abs(hm.sum(2)/7 - (s-b))))
            assert err < 2e-6
            if epoch == 0:
                assert not h.any()
                np.testing.assert_array_equal(s, b)
            errors = []
            for i, ident in enumerate(ids):
                row = paths.loc[(seed,arm,ident)]
                y, rival = int(row.class_id), int(row.rival)
                direct = 2*hm[i,y,rival]/7
                third = sum(hm[i,y,k]-hm[i,rival,k] for k in range(8) if k not in [y,rival])/7
                errors.extend([abs(direct-row.direct_pair_delta), abs(third-row.third_class_delta)])
                matrices[(seed,arm,ident)] = hm[i]
            checks.append(dict(seed=seed,arm=arm,epoch=epoch,score_flux_error=err,
                               path_error=max(errors),antisymmetry_error=float(abs(hm+hm.transpose(0,2,1)).max())))
    pd.DataFrame(checks).to_csv(HERE/'flux_replay.csv',index=False)
    return matrices


def render(changes, predictions, matrices):
    (HERE/'panels').mkdir(exist_ok=True)
    # Paired/marginal has no validation flips at any seed; do not invent them.
    choices = [(43,'global',outcome) for outcome in ['rescued','harmed','CC','WW']]
    choices.append((29,'B','rescued'))
    rows, edges = [], []
    valid = predictions[predictions.role.eq('validation')].set_index(['seed','arm','instance_id'])
    for seed, reference, outcome in choices:
        candidates = changes[changes.role.eq('validation') & changes.seed.eq(seed) & changes.reference.eq(reference) & changes.outcome.eq(outcome)].sort_values('instance_id')
        assert len(candidates)
        row = candidates.iloc[0]
        ident, y = row.instance_id, int(row.class_id)
        sg = {arm:score(valid.loc[[(seed,arm,ident)]])[0] for arm in run.ARMS}
        b = score(valid.loc[[(seed,'paired',ident)]], 'b')[0]
        rival = int(np.where(np.arange(8)==y, -np.inf, sg['paired']).argmax())
        oldrival = int(np.where(np.arange(8)==y, -np.inf, b if reference=='B' else sg[reference]).argmax())
        fig, ax = plt.subplots(2,3,figsize=(16,9),layout='constrained')
        dimensions = []
        for j,pol in enumerate(['vv','vh']):
            with Image.open(row[f'chip_path_{pol}']) as im:
                raw = np.array(im)
            padded, mask = native_image(row[f'chip_path_{pol}'])
            h,w = raw.shape
            top,left = (224-h)//2,(224-w)//2
            np.testing.assert_array_equal(np.rint(padded[0,top:top+h,left:left+w]*255).astype(np.uint8),raw)
            dimensions.append(f'{pol}:{w}x{h}')
            ax[j,0].imshow(raw,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            ax[j,0].set_title(f'{pol.upper()} original {w}x{h}')
            ax[j,1].imshow(padded[0],cmap='gray',vmin=0,vmax=1,interpolation='nearest')
            ax[j,1].set_title(f'Actual 224 pad; no resize/crop; footprint={mask.mean():.4f}')
        for offset,label,value,color in [(-.3,'B',b,'#aaaaaa'),(-.1,'global',sg['global'],'#526b83'),(.1,'marginal',sg['marginal'],'#be6472'),(.3,'paired',sg['paired'],'#24856d')]:
            ax[0,2].bar(np.arange(8)+offset,value,width=.19,label=label,color=color)
        ax[0,2].legend(fontsize=8)
        ax[0,2].set_title('Actual eight-class scores')
        contributions = {}
        for arm in run.ARMS:
            hm = matrices[(seed,arm,ident)]
            terms = np.array([2*hm[y,rival]/7 if k==rival else (hm[y,k]-hm[rival,k])/7 if k!=y else 0 for k in range(8)])
            contributions[arm] = terms
            actual = (sg[arm][y]-b[y])-(sg[arm][rival]-b[rival])
            assert abs(terms.sum()-actual)<2e-6
            for k in range(8):
                edges.append(dict(seed=seed,reference=reference,outcome=outcome,instance_id=ident,
                    arm=arm,true_class=y,fixed_rival=rival,via_class=k,
                    kind='direct' if k==rival else 'none' if k==y else 'third',contribution=terms[k],
                    total_actual=actual,closure_error=terms.sum()-actual))
        difference = contributions['paired']-(np.zeros(8) if reference=='B' else contributions[reference])
        ax[1,2].bar(np.arange(8),difference,color=['#be6472' if k==rival else '#526b83' for k in range(8)])
        ax[1,2].axhline(0,color='black',lw=.6)
        ax[1,2].set_title(f'Paired - {reference}: margin path at fixed rival {rival}\nred=direct pair; others=third-class paths')
        zero = int(row.epoch)==0
        note = 'ZERO EPOCH: paired equals B; not trained benefit' if zero else 'nonzero source-selected paired head'
        fig.suptitle(f'SOURCE VALIDATION / paired vs {reference} / {outcome} / seed{seed}\n{ident}; true={y}; {note}\n'
            f'reference winner={int(row.reference_prediction)} -> paired={int(row.prediction)}; '
            f'reference rival={oldrival}; paired rival={rival}; path sum={difference.sum():+.6f}',fontsize=11)
        path = HERE/'panels'/f'{reference}_{outcome}_{seed}.png'
        fig.savefig(path,dpi=100)
        plt.close(fig)
        rows.append(dict(seed=seed,reference=reference,outcome=outcome,instance_id=ident,class_id=y,
            paired_epoch=int(row.epoch),zero_epoch=zero,reference_prediction=int(row.reference_prediction),
            paired_prediction=int(row.prediction),old_rival=oldrival,fixed_rival=rival,
            direct_difference=difference[rival],third_difference=difference.sum()-difference[rival],
            total_difference=difference.sum(),dimensions=';'.join(dimensions),panel=str(path),
            image_vv=row.chip_path_vv,image_vh=row.chip_path_vh))
    pd.DataFrame(rows).to_csv(HERE/'review_cases.csv',index=False)
    pd.DataFrame(edges).to_csv(HERE/'case_paths.csv',index=False)
    return rows


def main():
    p = pd.read_csv(ROOT/'predictions.csv',dtype={'mmsi':str})
    assert set(p.role)=={'train','validation'} and set(p.fold)=={0}
    assert p.groupby(['role','seed','arm']).size().eq(p.groupby(['role','seed','arm']).instance_id.nunique()).all()
    values,changes,classes = [],[],[]
    for (role,seed),a in p[p.arm.eq('paired')].groupby(['role','seed']):
        for reference in ['B','marginal','global']:
            b = a if reference=='B' else p[p.role.eq(role)&p.seed.eq(seed)&p.arm.eq(reference)]
            metrics,g = compare(a,b,reference)
            meta = dict(role=role,seed=seed,reference=reference)
            values.append(dict(**meta,**metrics))
            changes.append(g)
            for c,sub in g.groupby('class_id'):
                classes.append(dict(**meta,class_id=c,n=len(sub),net=int(sub.net.sum()),
                    rescued=int(sub.outcome.eq('rescued').sum()),harmed=int(sub.outcome.eq('harmed').sum()),
                    recall=100*sub.prediction.eq(sub.class_id).mean(),delta_recall=100*sub.net.mean()))
    pd.DataFrame(values).to_csv(HERE/'paired_comparisons.csv',index=False)
    pd.DataFrame(classes).to_csv(HERE/'class_comparisons.csv',index=False)
    changes = pd.concat(changes,ignore_index=True)
    changes[['role','seed','reference','instance_id','class_id','epoch','prediction','reference_prediction','outcome','net']].to_csv(HERE/'prediction_changes.csv',index=False)
    matrices = fluxes(p)
    cases = render(changes,p,matrices)
    (HERE/'analysis_summary.json').write_text(json.dumps(dict(training_runs=0,test_evaluated=False,
        cases=len(cases),unique_case_objects=len({r['instance_id'] for r in cases}),images_viewed=False,
        unavailable='paired versus marginal rescued/harmed absent for all validation seeds',
        panel_selection='seed43 paired/global all four outcomes first ID; seed29 paired/B first rescue',
        selection_rationale='seed17 all zero; seed29 top1 identical across arms; seed43 only arm-to-arm flips',
        path_rival='paired highest non-true class fixed across all compared heads',pixels_resized=0,pixels_cropped=0),indent=2),encoding='utf-8')
    print(pd.DataFrame(values).to_string(index=False),flush=True)
    print(pd.DataFrame(cases).to_string(index=False),flush=True)


if __name__=='__main__':
    main()
