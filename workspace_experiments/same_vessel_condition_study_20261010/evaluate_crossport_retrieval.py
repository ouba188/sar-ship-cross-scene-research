"""Fixed cross-port/cross-vessel MAP@R probes; no fitting or model selection."""

from full_teacher_data import BASE, ROOT, SOURCE, PORTS, write
from case_support_completion import COLS, predict
from unified_head_expert import ExpertContext
from projection_residual_expert import discarded_coordinates
import numpy as np
import pandas as pd

OUT=BASE/'layered_evaluation_20261012/retrieval'
HEADROOM=BASE/'case_relation_headroom_20261012'
STUDENT=BASE/'case_relation_student_20261012'
SEEDS=(31415,27182)


def ap_at_r(relevant):
    r=int(np.sum(relevant))
    if not r:
        return np.nan
    top=np.asarray(relevant[:r],dtype=float)
    return float(np.sum(np.cumsum(top)/np.arange(1,r+1)*top)/r)


def allowed_candidates(query, gallery):
    return (gallery.port.to_numpy()!=query.port)&gallery.mmsi.notna().to_numpy()&(
        gallery.mmsi.to_numpy()!=query.mmsi)


def mutable_coordinates(extra,g,cached_projected=None):
    model,stats=g['model'],g['stats']
    ec=np.asarray(extra,dtype=float)-model['pca_ssl'].mean_
    projected=((ec@np.asarray(model['pca_ssl'].components_,dtype=float).T).astype(np.float32)
               if cached_projected is None else np.asarray(cached_projected,dtype=np.float32))
    # The full map stacks these FP32 PCA coordinates with FP64 fixed coordinates first.
    retained=(projected.astype(float)-model['expert_center'][128:192])/model['expert_scale'][128:192]
    comp=(discarded_coordinates(extra,model['pca_ssl'])-stats['discarded_center'][2048:])/stats['discarded_scale'][2048:]
    return np.c_[retained/np.sqrt(192)/np.sqrt(2),comp/np.sqrt(2560)/np.sqrt(2)]


def evaluate(qphi,gphi,qmeta,gmeta,config,correction=None):
    order=np.argsort(gmeta.obs_id.to_numpy(),kind='stable')
    gphi,gmeta=gphi[order],gmeta.iloc[order].reset_index(drop=True)
    if correction is not None:
        correction=correction[order]
    rows=[]
    qnorm=np.linalg.norm(qphi,axis=1);gnorm=np.linalg.norm(gphi,axis=1)
    assert np.all(qnorm>0) and np.all(gnorm>0)
    gram=qphi@gphi.T
    # Only the mutable block changes, so the untouched dot products can be reused.
    active=np.arange(len(COLS)) if config['view']=='r18_coordinates' else COLS
    if correction is not None:
        qx,gx=qphi[:,active],gphi[:,active]
        suggested=gx+correction
        fixed_dot=gram-qx@gx.T
        fixed_norm=np.maximum(np.square(gphi).sum(1)-np.square(gx).sum(1),0)
    checked=False
    for j,row in qmeta.iterrows():
        record=dict(**config,obs_id=int(row.obs_id),port=row.port,mmsi=row.mmsi,class_id=int(row.class_id))
        if pd.isna(row.mmsi):
            rows.append(dict(**record,status='unknown_query_identity',n_candidates=0,r=0,map_r=np.nan,recall1=np.nan))
            continue
        candidates=np.flatnonzero(allowed_candidates(row,gmeta))
        if not len(candidates):
            rows.append(dict(**record,status='no_candidates',n_candidates=0,r=0,map_r=np.nan,recall1=np.nan))
            continue
        if correction is None:
            similarity=gram[j,candidates]/(qnorm[j]*gnorm[candidates])
        else:
            raw,candidate=gx[candidates],suggested[candidates]
            rc,tc=raw.mean(0),candidate.mean(0)
            re=np.square(raw-rc).sum(1).mean();te=np.square(candidate-tc).sum(1).mean()
            assert te>0
            revised=(candidate-tc)*np.sqrt(re/te)+rc
            similarity=(fixed_dot[j,candidates]+qx[j]@revised.T)/(
                qnorm[j]*np.sqrt(fixed_norm[candidates]+np.square(revised).sum(1)))
            if not checked:
                explicit=gphi[candidates].copy();explicit[:,active]=revised
                direct=(qphi[j]@explicit.T)/(qnorm[j]*np.linalg.norm(explicit,axis=1))
                np.testing.assert_allclose(similarity,direct,atol=1e-12,rtol=0)
                checked=True
        ranking=candidates[np.argsort(-similarity,kind='stable')]
        rel=gmeta.iloc[ranking].class_id.to_numpy()==row.class_id
        r=int(rel.sum())
        rows.append(dict(**record,status='ok' if r else 'no_crossport_class_positive',
            n_candidates=len(candidates),r=r,map_r=ap_at_r(rel),recall1=float(rel[0]) if r else np.nan))
    return rows


def test():
    np.testing.assert_allclose(ap_at_r([True,False,True,True]),5/9)
    assert ap_at_r([True,True,False])==1.
    assert np.isnan(ap_at_r([False,False]))
    gallery=pd.DataFrame(dict(port=['a','b','b','b'],mmsi=[2.,1.,2.,np.nan]))
    query=pd.Series(dict(port='a',mmsi=1.))
    np.testing.assert_array_equal(allowed_candidates(query,gallery),[False,False,True,False])


def main():
    test();OUT.mkdir(parents=True,exist_ok=True)
    write(OUT/'PROTOCOL.json',dict(
        metric='AP@R=sum_{k<=R}(precision@k * relevance@k)/R; R=eligible same-class gallery count',
        reference='Musgrave official pytorch-metric-learning accuracy_calculator mean_average_precision(at_r=True)',
        query_gallery='Same fixed823 source-held observations; actualtargetqueries additionally use that same823 sourceheld gallery',
        eligible='Different actual port AND different known MMSI; unknown identities excluded; R=0 queries reported, not scored as zero',
        candidate_distribution='All eligible fixed rows, not rebalanced or resampled by arm; same-vessel gallery repeats remain disclosed',
        aggregation='Within each actual port/class average queries, then classes within port, then ports; micro separately',
        views='Primary full expert phi WITHOUT constant bias; secondary R18-dependent576 coordinates; cosine distance, stable obs-id tie order',
        student='Task/relation/readout maps both query and gallery; two true training seeds, three geometry contexts, not actual queryports',
        completion='Raw queries, corrected eligible gallery only; reproduce original eligible-gallery centroid/energy matching, no labels in predictor',
        scope='Gallery retrieval diagnostic, not the deployed64-support classifier or added budget labels',
        temperature='Not used for retrieval',new_fits=0,parameters_selected=False,
        limitations='Existing repeated development cohorts; historical source encoder/geometry exposure; exploratory metric establishment'))
    meta=pd.read_csv(BASE/'observations.csv')
    manifest=pd.read_csv(SOURCE/'manifest.csv',usecols=['class_id','mmsi','port'])
    records,parity=[],[]
    for port in PORTS:
        with np.load(HEADROOM/port/'maps.npz') as z:
            ids,raw,prior=(z[k] for k in ('obs_ids','single','prior'))
        held=np.arange(1296,len(ids));assert len(held)==823
        mm=meta.iloc[ids].reset_index(drop=True)
        hm=mm.iloc[held].reset_index(drop=True)
        context=ExpertContext(ROOT/'learned_frequency_residual_20261009/tasks/self24ports'/port)
        source_rows=np.load(context.parent/'source_rows.npy')
        pos=pd.Index(source_rows).get_indexer(mm.manifest_row)
        assert (pos>=0).all()
        initial_extra=np.load(ROOT/'evidence_aware_handoff_20261008/source_features'/port/'features.npy',mmap_mode='r')[pos]
        cached_bank=np.load(context.parent/'source_bank.npy',mmap_mode='r')
        initial_block=mutable_coordinates(initial_extra,context.geometry,cached_bank[pos,128:192])
        np.testing.assert_allclose(initial_block,raw[:,COLS],atol=1e-12,rtol=0)
        fresh=mutable_coordinates(initial_extra,context.geometry)
        parity.append(dict(context=port,initial_block_max_error=float(np.max(abs(initial_block-raw[:,COLS]))),
            historical_cached_vs_fresh_projection_gap=float(np.max(abs(fresh-initial_block)))))
        for seed in SEEDS:
            for arm in ('task','relation','readout'):
                path=(STUDENT/port/str(seed)/arm if arm!='readout' else
                      BASE/'case_relation_readout_repair_20261012'/port/str(seed)/arm)
                phi=raw.copy();phi[:,COLS]=mutable_coordinates(np.load(path/'extra.npy'),context.geometry)
                # Replaying one saved classifier checks that reconstruction means the same model.
                from unified_head_core import score
                with np.load(HEADROOM/port/'31415_0_single_native.npz') as z:
                    s=pd.Index(ids).get_indexer(z['support_obs_ids']);q=pd.Index(ids).get_indexer(z['query_obs_ids']);sy=z['support_labels']
                expected=score(phi@phi[s].T,prior,s,sy)[0][q]
                with np.load(path/'31415_0.npz') as z:
                    np.testing.assert_allclose(expected,z['scores'],atol=1e-10,rtol=0)
                for view,columns in [('full_expert',np.arange(phi.shape[1]-1)),('r18_coordinates',COLS)]:
                    records.extend(evaluate(phi[held][:,columns],phi[held][:,columns],hm,hm,
                        dict(family='source_student',context=port,training_seed=seed,arm=arm,view=view)))
                print('RETRIEVED student',port,seed,arm,flush=True)
        models={}
        for arm,path in [('image_history',BASE/'case_support_completion_20261012'/port/'final_model.npz'),
                         ('class_common',BASE/'case_completion_class_control_20261012'/port/'final_model.npz')]:
            with np.load(path) as z:models[arm]={k:z[k] for k in z.files}
        with np.load(BASE/'full_teacher_integration_20261012'/port/'no_training/predictions.npz') as z:
            target_phi,target_rows=z['phi'],z['rows']
        tm=manifest.iloc[target_rows].reset_index(drop=True);tm['obs_id']=target_rows
        for family,qphi,qmeta in [('source_completion',raw[held],hm),('target_retrieval',target_phi,tm)]:
            for arm in ('raw','image_history','class_common'):
                correction=None if arm=='raw' else predict(models[arm],raw[held][:,COLS],np.zeros(len(held),int),'image_history')
                for view,columns in [('full_expert',np.arange(raw.shape[1]-1)),('r18_coordinates',COLS)]:
                    records.extend(evaluate(qphi[:,columns],raw[held][:,columns],qmeta,hm,
                        dict(family=family,context=port,training_seed=0,arm=arm,view=view),correction))
                print('RETRIEVED',family,port,arm,flush=True)
        del context
    frame=pd.DataFrame(records);frame.to_csv(OUT/'query_metrics.csv',index=False)
    keys=['family','context','training_seed','arm','view']
    coverage=frame.groupby(keys+['status']).agg(queries=('obs_id','size')).reset_index()
    coverage.to_csv(OUT/'coverage.csv',index=False)
    ok=frame[frame.status=='ok']
    cells=ok.groupby(keys+['port','class_id']).agg(map_r=('map_r','mean'),recall1=('recall1','mean'),
        queries=('obs_id','size'),ships=('mmsi','nunique'),candidates_mean=('n_candidates','mean'),r_mean=('r','mean')).reset_index()
    cells.to_csv(OUT/'port_class.csv',index=False)
    ports=cells.groupby(keys+['port'])[['map_r','recall1']].mean().reset_index()
    ports.to_csv(OUT/'actual_ports.csv',index=False)
    context_summary=ports.groupby(keys)[['map_r','recall1']].mean().reset_index()
    micro=ok.groupby(keys)[['map_r','recall1']].mean().add_prefix('micro_').reset_index()
    context_summary=context_summary.merge(micro,on=keys)
    context_summary.to_csv(OUT/'contexts.csv',index=False)
    summary=context_summary.groupby(['family','arm','view'])[['map_r','recall1','micro_map_r','micro_recall1']].mean()
    summary.to_csv(OUT/'summary.csv')
    write(OUT/'REPLAY.json',dict(initial_blocks=parity,student_saved_readouts_replayed=18,absolute_tolerance=1e-10))
    print(summary.to_string(),flush=True)


if __name__=='__main__':
    main()
