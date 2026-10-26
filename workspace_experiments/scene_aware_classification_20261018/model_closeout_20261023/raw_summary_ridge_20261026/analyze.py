"""Paired cohort uncertainty and native image evidence for the ridge readout test."""
import os
from pathlib import Path
HERE = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(HERE/'.mplcache')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import run
from analyze_transfer import input_records, CLASSES
base = run.base


def main():
    out = HERE/'analysis'
    out.mkdir(exist_ok=True)
    paired, classes, intervals, groups, cases, predictions, diagnostics = [], [], [], [], [], [], []
    for name in ('oss1', 'oss2'):
        root, cross, source, target, raw, eraw, local, elocal = base.dataset(name)
        table = base.tr.read_table(cross/'target_table.csv', labels=True)
        np.testing.assert_array_equal(target.obs_id, table.obs_id)
        y = table.class_id.to_numpy(int)
        z = base.tr.load_npz(HERE/name/'cross_predictions.npz')
        np.testing.assert_array_equal(z['obs_id'], table.obs_id)
        old, new = z['raw_ridge'], z['raw_summary_ridge']
        bc, ac = old.argmax(1)==y, new.argmax(1)==y
        d = ac.astype(float)-bc
        paired.append(dict(source=name, n=len(y), rescued=int((ac&~bc).sum()), harmed=int((bc&~ac).sum()),
            wrong_to_wrong=int((~ac&~bc&(old.argmax(1)!=new.argmax(1))).sum()),
            OA_pp=100*d.mean(), BA_pp=100*np.mean([d[y==c].mean() for c in range(4)])))
        p = table[['obs_id','mmsi','pass_group','class_id']].copy()
        p['source'], p['before'], p['after'] = name, old.argmax(1), new.argmax(1)
        predictions.append(p)
        for c in range(4):
            ids = y==c
            classes.append(dict(source=name, class_name=CLASSES[c], n=int(ids.sum()),
                before_recall=100*bc[ids].mean(), after_recall=100*ac[ids].mean(),
                rescue=int((ac&~bc&ids).sum()), harm=int((bc&~ac&ids).sum())))
        for field in ('pass_group', 'mmsi'):
            ix = list(table.groupby(field).indices.values())
            count = np.array([np.bincount(y[t], minlength=4) for t in ix])
            sums = np.array([np.bincount(y[t], weights=d[t], minlength=4) for t in ix])
            draw = np.random.default_rng(1026).integers(len(ix), size=(4000,len(ix)))
            ct, sm = count[draw].sum(1), sums[draw].sum(1)
            valid = (ct>0).all(1)
            oa, ba = 100*sm.sum(1)/ct.sum(1), 100*(sm[valid]/ct[valid]).mean(1)
            intervals.append(dict(source=name, cluster=field, clusters=len(ix),
                OA_low=np.quantile(oa,.025), OA_high=np.quantile(oa,.975),
                BA_low=np.quantile(ba,.025), BA_high=np.quantile(ba,.975), complete_draws=int(valid.sum())))
        for group, ix in table.groupby('pass_group').indices.items():
            groups.append(dict(source=name, pass_group=group, n=len(ix), OA_pp=100*d[ix].mean()))

        x = base.representations(raw, local)['regional']
        xe = base.representations(eraw, elocal)['regional']
        aux, eaux = run.prior.summary_inputs(root, cross, source, target)
        sc, sa = base.tr.load_npz(HERE/name/'raw_scaler.npz'), base.tr.load_npz(HERE/name/'summary_scaler.npz')
        x, xe = base.transform(x, sc), base.transform(xe, sc)
        eaux = base.transform(eaux, sa)
        before = base.tr.load_npz(HERE/name/'raw/ridge_model.npz')
        after = base.tr.load_npz(HERE/name/'raw_summary/ridge_model.npz')
        dim = x.shape[1]
        response = eaux[:,:72] @ after['coef'][:,dim:dim+72].T
        geometry = eaux[:,72:] @ after['coef'][:,dim+72:].T
        raw_change = xe @ (after['coef'][:,:dim]-before['coef']).T
        bias_change = after['intercept']-before['intercept']
        # Cached ridge uses float32 GEMM; splitting its sum changes rounding.
        closure = float(abs(raw_change+response+geometry+bias_change-(new-old)).max())
        np.testing.assert_allclose(raw_change+response+geometry+bias_change, new-old, atol=2e-5, rtol=1e-5)
        # Diagnostic switches keep the fitted parameters fixed; these are not new candidates.
        for label, score in [('full',new),('remove_response_fixed_head',new-response),
                             ('remove_geometry_fixed_head',new-geometry),('remove_both_fixed_head',new-response-geometry)]:
            diagnostics.append(dict(source=name, diagnostic=label, **base.classification_metrics(y,score)))

        combined = pd.read_csv(cross/'table.csv', dtype={'mmsi':str})
        records = input_records(combined, base.tr.read_json(cross/'protocol.json'))
        qs = combined[combined.dataset_side.eq('target')]
        ss = combined[combined.dataset_side.eq('source')]
        qix = pd.Index(qs.original_obs_id).get_indexer(table.obs_id)
        six = pd.Index(ss.original_obs_id).get_indexer(source.obs_id)
        assert (qix>=0).all() and (six>=0).all()
        query_rows, source_rows = qs.index.to_numpy()[qix], ss.index.to_numpy()[six]
        fit = np.flatnonzero(source.role.eq('fit_train'))
        for state, mask in dict(rescue=ac&~bc,harm=bc&~ac,CC=ac&bc,WW=~ac&~bc).items():
            ids = np.flatnonzero(mask)
            if not len(ids):
                continue
            focus = ids[y[ids]==1]
            if name=='oss2' and state in ('rescue','harm') and len(focus):
                ids = focus
            j = int(ids[np.argmin(table.iloc[ids].obs_id.to_numpy())])
            true = int(y[j])
            rival = int(old[j].argmax()) if not bc[j] else int(new[j].argmax())
            if rival==true:
                rival = int(np.where(np.arange(4)==true,-np.inf,old[j]).argmax())
            neighbors=[]
            for c in (true,rival):
                candidates=fit[source.iloc[fit].class_id.to_numpy(int)==c]
                dist=((x[candidates]-xe[j])**2).mean(1)
                neighbors.extend(candidates[np.argsort(dist,kind='stable')[:2]].tolist())
            rows=[int(query_rows[j]), *source_rows[neighbors].astype(int).tolist()]
            fig, axes=plt.subplots(2,5,figsize=(15,6.7))
            prov=[]
            for col,row in enumerate(rows):
                record=records[row]
                for pol,v in enumerate(base.tr.extract.prepare(record)):
                    ax=axes[pol,col]
                    if v is not None:
                        ax.imshow(v['pixels'],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                        if v['masks'][0].any():
                            ax.contour(v['masks'][0],levels=[.5],colors=['#20a780'],linewidths=.6)
                        prov.append(dict(row=row,pol=('VV','VH')[pol],path=record[('vv','vh')[pol]]['image'],placement=v['placement']))
                    ax.set_title(f'{("VV","VH")[pol]} {("QUERY","TRUE TRAIN 1","TRUE TRAIN 2","RIVAL TRAIN 1","RIVAL TRAIN 2")[col]}\nobs {combined.iloc[row].original_obs_id}',fontsize=9)
                    ax.set(xticks=[],yticks=[])
            def pair(v):
                return float(v[true]-v[rival])
            terms=dict(raw_refit=pair(raw_change[j]), response_summary=pair(response[j]),
                       geometry_quality=pair(geometry[j]), intercept=pair(bias_change))
            typed=(eaux[j,:72]*(after['coef'][true,dim:dim+72]-after['coef'][rival,dim:dim+72])).reshape(2,3,4,3)
            region_terms={f'{pol}_{role}':float(typed[ip,ir].sum())
                for ip,pol in enumerate(('VV','VH')) for ir,role in enumerate(('body','end','side'))}
            statistic_terms={stat:float(typed[:,:,:,it].sum())
                for it,stat in enumerate(('mean','half_high_minus_low','half_high_plus_low'))}
            np.testing.assert_allclose(sum(region_terms.values()),terms['response_summary'],atol=2e-5)
            mb, ma=pair(old[j]), pair(new[j])
            np.testing.assert_allclose(sum(terms.values()),ma-mb,atol=2e-5)
            filename=f'{name}_{state}_{int(table.iloc[j].obs_id)}.png'
            fig.suptitle(f'{cross.name}: {state}; true {CLASSES[true]}; ridge {CLASSES[old[j].argmax()]} -> +summary {CLASSES[new[j].argmax()]}\nFixed pair {CLASSES[true]} vs {CLASSES[rival]}: {mb:+.4f} -> {ma:+.4f}')
            fig.text(.02,.02,'Native160 crop/pad. Nearest source-fit examples for inspection only; ridge does not retrieve references. No pixel intervention.',fontsize=8)
            fig.subplots_adjust(top=.80,bottom=.09,hspace=.2,wspace=.1)
            fig.savefig(out/filename,dpi=115)
            plt.close(fig)
            cases.append(dict(source=name,status=state,obs_id=int(table.iloc[j].obs_id),true=true,rival=rival,
                before_pred=int(old[j].argmax()),after_pred=int(new[j].argmax()),before_margin=mb,after_margin=ma,
                decomposition=terms,summary_by_region=region_terms,summary_by_statistic=statistic_terms,
                decomposition_closure_max=closure,image=filename,source_fit_examples=source.iloc[neighbors].obs_id.to_numpy(int).tolist(),
                example_selection='two same/rival class nearest by source-standardized raw regional squared distance; not runtime supports',
                provenance=prov))
    for label, rows in [('paired',paired),('classwise',classes),('uncertainty',intervals),('passwise',groups),('fixed_head_diagnostics',diagnostics)]:
        pd.DataFrame(rows).to_csv(out/f'{label}.csv',index=False)
    pd.concat(predictions).to_csv(out/'predictions.csv.gz',index=False,compression='gzip')
    base.write_json(out/'cases.json',cases)
    base.write_json(out/'receipt.json',dict(panels=len(cases),raw_review_pending=True,
        decomposition='exact linear fixed-class score difference, not physical attribution',
        bootstrap='4000 pass or MMSI cluster draws separately, fixed trained models; no source resampling'))
    print(pd.DataFrame(paired).to_string(index=False))
    print(pd.DataFrame(classes).to_string(index=False))
    print(pd.DataFrame(intervals).to_string(index=False))


if __name__=='__main__':
    main()
