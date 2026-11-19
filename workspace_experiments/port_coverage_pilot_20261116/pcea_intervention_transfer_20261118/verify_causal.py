"""Bounded real-source checks: wiring, live references, gradients and two steps."""
import gc
import json
import time
from pathlib import Path
import torch
import numpy as np
import causal_run as run


def difference(a,b):
    return float((a-b).abs().max())


def gallery(inputs,ids,trace,before,after,legacy,donors,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    refs=inputs.groups['references']
    records=[]
    for j,idx in enumerate(ids):
        row=inputs.rows[idx]
        label=int(inputs.y[idx])
        rival=int(np.where(np.arange(8)==label,-np.inf,after[j]).argmax())
        same=int(refs[np.flatnonzero(inputs.y[refs]==label)[0]])
        other=int(refs[np.flatnonzero(inputs.y[refs]==rival)[0]])
        record=dict(index=int(idx),id=row['id'],label=label,
            original_paths={p:row[p]['image'] for p in ('vv','vh')},donor_index=int(ids[donors[j]]),
            references=[same,other],legacy=legacy[j].tolist(),split_initial=before[j].tolist(),
            after_two_checks=after[j].tolist())
        records.append(record)
        fig,axes=plt.subplots(2,5,figsize=(13,7),layout='constrained')
        for p,pol in enumerate(('vv','vh')):
            for col,source in ((0,idx),(3,same),(4,other)):
                with Image.open(inputs.rows[source][pol]['image']) as im:
                    axes[p,col].imshow(np.asarray(im),cmap='gray',interpolation='nearest')
            energies=[np.log1p(trace[key][j,p].square().mean(0).sqrt().numpy()) for key in ('causal','bias')]
            for col,energy in enumerate(energies,1):
                axes[p,col].imshow(energy,cmap='magma',interpolation='nearest',vmin=0,
                    vmax=max(float(a.max()) for a in energies))
            for col,title in enumerate(('Original PNG','Retained log-energy','Complement log-energy','Same-class ref','Rival-class ref')):
                axes[p,col].set_title(pol+' '+title,fontsize=9)
                axes[p,col].axis('off')
        fig.suptitle(f"Source {row['id']} | label {label} | legacy {legacy[j].argmax()} / initial {before[j].argmax()} / 2-step {after[j].argmax()}\nUntrained split initialization; energy is NOT causal attribution")
        fig.savefig(out/f'case_{j:02d}_local.png',dpi=130)
        plt.close(fig)
    (out/'cases_local.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    tick=time.perf_counter()
    torch.set_num_threads(4)
    out=run.OUT
    out.mkdir(exist_ok=True)
    inputs,fit,_=run.base.runtime('unified')
    bank=torch.load(run.ROOT/'unified_20261119/verification_bank_local.pt',map_location='cpu',weights_only=True)
    ids=bank['ids'][[0,1,12,13]].numpy()
    assert np.isin(ids,fit).all() and not np.isin(ids,inputs.groups['references']).any()
    data=inputs.data(ids,'small')
    rd=inputs.data(inputs.groups['references'],'small')
    rows=[inputs.rows[i] for i in ids]
    old=run.base.model('unified').eval()
    with torch.no_grad():
        ref=old.reference(rd)
        legacy,_=old(data,rows,ref,inputs.reference_rows)
        legacy=legacy.cpu()
    del old,ref
    gc.collect()
    torch.cuda.empty_cache()
    net=run.make_model(1.).eval()
    with torch.no_grad():
        ref=net.reference(rd)
        replay,_=net(data,rows,ref,inputs.reference_rows)
    error=difference(replay.cpu(),legacy)
    assert error < 1e-4,error
    net.disentangler.keep=128
    run.set_source_prior(net,inputs,np.setdiff1d(fit,inputs.groups['references']))
    net.capture_split=True
    with torch.no_grad():
        ref=net.reference(rd)
        ref_desc=ref['desc'].clone()
        source=net.exchange.source_raw.clone()
        clean,_=net(data,rows,ref,inputs.reference_rows)
        trace=dict(net.split_trace)
        assert difference(trace['raw'],trace['causal']+trace['bias']) == 0
        assert float((trace['causal']*trace['bias']).abs().max()) == 0
        def change_discarded(module,args,result):
            x=result.reshape(-1,2,256,40,40)
            excluded=1-net.disentangler.mask()[None,:,:,None,None]
            return (x+.37*excluded).flatten(0,1)
        handle=net.net.base.backbone.model.layer1.register_forward_hook(change_discarded,prepend=True)
        try:
            changed_ref=net.reference(rd)
            reference_error=difference(changed_ref['desc'],ref_desc)
            pcea_error=difference(net.exchange.source_raw,source)
            changed,_=net(data,rows,changed_ref,inputs.reference_rows)
            bypass_error=difference(changed,clean)
            # Negative control: excluded input changes must reach the bias branch.
            biased_changed,_=net.forward_variant(data,rows,changed_ref,inputs.reference_rows,'bias')
        finally:
            handle.remove()
        ref=net.reference(rd)
        biased,_=net.forward_variant(data,rows,ref,inputs.reference_rows,'bias')
        bias_effect=difference(biased_changed,biased)
        assert max(reference_error,pcea_error,bypass_error) < 1e-5
        assert bias_effect > 1e-7,bias_effect
        donor,eligible=run.donor_permutation(data['available'],torch.Generator().manual_seed(17))
        mixed,_=net.forward_variant(data,rows,ref,inputs.reference_rows,'swap',donor)
        swap_effect=difference(mixed,clean)
        assert swap_effect > 1e-7
        net.train()
        ref=net.reference(rd)
        train_z,_=net(data,rows,ref,inputs.reference_rows)
        mode_error=difference(train_z,clean)
        assert mode_error < 1e-4,mode_error
    net.capture_split=False
    groups=net.optimizer_groups()
    assigned=[id(p) for g in groups for p in g['params']]
    assert len(assigned)==len(set(assigned))
    assert set(assigned)=={id(p) for p in net.parameters() if p.requires_grad}
    optimizer=torch.optim.AdamW(groups,weight_decay=.01)
    state0=net.disentangler.scores.detach().clone()
    records=[]
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        _,stats=run.backward(net,inputs,ids,step*3,torch.Generator().manual_seed(17+step))
        norms={g['name']:sum(float(p.grad.square().sum()) for p in g['params'] if p.grad is not None)**.5 for g in groups}
        assert all(np.isfinite(v) and v > 0 for v in norms.values()),norms
        assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g['params']],5.,error_if_nonfinite=True)
        optimizer.step()
        records.append(dict(**stats,gradient_norms=norms))
        print('smoke step',step+1,stats,flush=True)
    delta=difference(net.disentangler.scores,state0)
    assert delta > 0
    net.eval()
    with torch.no_grad():
        ref=net.reference(rd)
        after,_=net(data,rows,ref,inputs.reference_rows)
    saved=out/'smoke_checkpoint_local.pt'
    torch.save(net.state_dict(),saved)
    net.load_state_dict(torch.load(saved,map_location='cuda',weights_only=True),strict=True)
    with torch.no_grad():
        ref=net.reference(rd)
        restored,_=net(data,rows,ref,inputs.reference_rows)
    checkpoint_error=difference(restored,after)
    assert checkpoint_error < 1e-6
    gallery(inputs,ids,trace,clean.cpu().numpy(),after.cpu().numpy(),legacy.numpy(),donor,out)
    result=dict(passed=True,source_samples=len(ids),reference_count=len(inputs.groups['references']),
        optimizer_steps=2,not_a_performance_experiment=True,external_evaluation=False,
        full_capacity_legacy_replay=error,excluded_query_input_logit_effect=bypass_error,
        excluded_reference_desc_effect=reference_error,excluded_pcea_source_effect=pcea_error,
        excluded_input_bias_branch_effect=bias_effect,swap_logit_effect=swap_effect,
        train_eval_error=mode_error,checkpoint_error=checkpoint_error,score_update=delta,
        retained_channels_per_polar=net.disentangler.keep,total_channels_per_polar=256,
        source_prior=net.source_prior.cpu().tolist(),steps=records,
        peak_cuda_gib=torch.cuda.max_memory_allocated()/1024**3,seconds=time.perf_counter()-tick)
    (out/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
