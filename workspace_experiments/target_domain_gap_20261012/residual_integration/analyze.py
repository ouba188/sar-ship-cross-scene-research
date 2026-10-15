"""Cache-only accounting of decision changes and support-draw sensitivity."""

from experiment import *


def main():
    groups,classes,transmissions,pair_movements=[],[],[],[]
    for folder in FOLDERS:
        info=json.loads((folder/'PREPARED.json').read_text())
        task=info['task']
        z=load_npz(OUT/'target'/task/'predictions.npz')
        path=SOURCE/'manifest.csv' if info['dataset']=='self24ports' else ROOT/DATASETS['oss2'][0]/'target_manifest.csv'
        y=pd.read_csv(path,usecols=['class_id']).class_id.to_numpy(int)[z['rows']]
        for draw in range(5):
            mask=z['eval_masks'][draw]
            yy=y[mask]
            counts=np.bincount(yy,minlength=8)
            p0=z['full'][draw,0,mask]
            e0=z['expert'][draw,0,mask]
            b0=2*p0-e0
            for ai,arm in enumerate(ARMS[1:],1):
                p=z['full'][draw,ai,mask]
                ep=z['expert'][draw,ai,mask]
                np.testing.assert_allclose(2*p-ep,b0,atol=1e-14,rtol=1e-13)
                before,after=p0.argmax(1),p.argmax(1)
                expert0,expert1=e0.argmax(1),ep.argmax(1)
                if task=='oss2' and arm=='retain_full':
                    old_prior=z['expert_prior'][draw,0,mask]
                    new_prior=z['expert_prior'][draw,ai,mask]
                    log_delta=np.log(new_prior)-np.log(old_prior)
                    score_margin_delta=log_delta[:,4]-log_delta[:,0]
                    prob_margin_delta=(p[:,4]-p[:,0])-(p0[:,4]-p0[:,0])
                    for cls in (0,4):
                        member=yy==cls
                        pair_movements.append(dict(task=task,draw=draw,class_id=cls,
                            n=int(member.sum()),
                            prior_container_vs_bulk_delta=float(score_margin_delta[member].mean()),
                            prior_container_vs_bulk_sd=float(score_margin_delta[member].std()),
                            prior_fraction_toward_container=float((score_margin_delta[member]>0).mean()),
                            full_container_vs_bulk_delta=float(prob_margin_delta[member].mean()),
                            full_fraction_toward_container=float((prob_margin_delta[member]>0).mean())))
                rescue=(before!=yy)&(after==yy)
                harm=(before==yy)&(after!=yy)
                net=rescue.astype(int)-harm.astype(int)
                meta=dict(task=task,draw=draw,arm=arm)
                for changed in (False,True):
                    member=(expert0!=expert1)==changed
                    groups.append(dict(**meta,expert_argmax_changed=changed,n=int(member.sum()),
                        rescues=int(rescue[member].sum()),harms=int(harm[member].sum()),
                        oa_contribution=float(100*net[member].sum()/len(yy)),
                        ba_contribution=float(100*np.sum(net[member]/counts[yy[member]])/8)))
                for c in range(8):
                    member=yy==c
                    classes.append(dict(**meta,class_id=c,n=int(member.sum()),
                        recall_delta=float(100*net[member].mean()),
                        rescues=int(rescue[member].sum()),harms=int(harm[member].sum())))
                expert_rescue=(expert0!=yy)&(expert1==yy)
                expert_harm=(expert0==yy)&(expert1!=yy)
                transmissions.append(dict(**meta,
                    expert_rescues=int(expert_rescue.sum()),expert_harms=int(expert_harm.sum()),
                    expert_rescues_final_still_wrong=int((expert_rescue&(after!=yy)).sum()),
                    expert_harms_final_still_correct=int((expert_harm&(after==yy)).sum()),
                    full_rescues_expert_new_wrong=int((rescue&(expert1!=yy)).sum()),
                    full_harms_expert_new_correct=int((harm&(expert1==yy)).sum()),
                    full_rescues_base_correct=int((rescue&(b0.argmax(1)==yy)).sum()),
                    full_harms_base_correct=int((harm&(b0.argmax(1)==yy)).sum())))
    pd.DataFrame(groups).to_csv(OUT/'fusion_groups.csv',index=False)
    pd.DataFrame(classes).to_csv(OUT/'class_changes.csv',index=False)
    pd.DataFrame(transmissions).to_csv(OUT/'rescue_transmission.csv',index=False)
    pd.DataFrame(pair_movements).to_csv(OUT/'oss2_pair_movements.csv',index=False)
    contrast=pd.read_csv(OUT/'target_contrasts.csv')
    rows=[]
    for (task,scope,stage,arm),f in contrast.groupby(['task','scope','stage','arm']):
        row=dict(task=task,scope=scope,stage=stage,arm=arm)
        for m in ('oa','ba','nll','brier'):
            x=f['delta_'+m].to_numpy()
            row.update({m+'_mean':float(x.mean()),m+'_sd':float(x.std(ddof=1)),
                        m+'_positive':int((x>1e-10).sum()),m+'_negative':int((x<-1e-10).sum()),
                        m+'_leave_one_draw_min':float(np.min((x.sum()-x)/(len(x)-1))),
                        m+'_leave_one_draw_max':float(np.max((x.sum()-x)/(len(x)-1)))})
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/'draw_sensitivity.csv',index=False)
    for name,frame in [('GROUPS',pd.DataFrame(groups)),('TRANSMISSION',pd.DataFrame(transmissions)),
                       ('CLASSES',pd.DataFrame(classes))]:
        subset=frame[(frame.task=='oss2')&(frame.arm=='retain_full')]
        keys=['expert_argmax_changed'] if name=='GROUPS' else (['class_id'] if name=='CLASSES' else ['task'])
        print(name,flush=True)
        print(subset.drop(columns=['draw','arm','task'] if name!='TRANSMISSION' else ['draw','arm'])
              .groupby(keys).mean(numeric_only=True).to_string(),flush=True)
    write(OUT/'ANALYSIS_COMPLETE.json',dict(frozen_base_replayed=True,new_training=False,
          sensitivity='Five overlapping support draws, not independent datasets or confidence bounds'))


if __name__=='__main__':
    main()
