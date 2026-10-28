"""Fixed-weight pathway and ReLU transmission diagnostics, not new fits."""
import gc
import importlib.util
from run import ROOT, PREVIOUS, RELEASE, ARMS, SEEDS, Cache, construct, setup, read_manifest, pair_mask, write, np, torch

spec = importlib.util.spec_from_file_location('previous_path_diagnosis', PREVIOUS/'diagnose.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
metric, margin = old.metric, old.margin


def entry(score, label, rival):
    value = score.detach().cpu().numpy()[0]
    return dict(pred=int(value.argmax()), top1_margin=float(margin(value[None], np.array([label]))[0]),
                fixed_margin=float(value[label]-value[rival]))


@torch.no_grad()
def main():
    setup('cuda')
    data = np.load(ROOT/'predictions.npz'); y = data['labels']; scores = data['scores']
    records = read_manifest(RELEASE/'manifests/development.json')
    refs = read_manifest(RELEASE/'manifests/references.json')
    cache = Cache('dev'); bank = Cache('refs').get(np.arange(len(refs)))
    state = torch.load(RELEASE/'weights/teacher.pt', weights_only=True, map_location='cpu')['state']
    output, cases, transmissions, saved = [], [], [], {}
    for arm in ('preadd','postadd','stage'):
        ai = list(ARMS).index(arm)
        for si, seed in enumerate(SEEDS):
            model = construct(state, ARMS[arm], seed)
            checkpoint = (ROOT/'checkpoints'/f'{arm}_{seed}.pt' if arm != 'stage' else
                          PREVIOUS.parent/'shallow_m2_writeback_20261028'/'checkpoints'/f'layer2_{seed}.pt')
            model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location='cpu')['state'], strict=False)
            for route in ('backbone_only','readout_only') if arm != 'stage' else ():
                chunks = []
                for first in range(0,len(y),16):
                    ix = np.arange(first,min(first+16,len(y)))
                    chunks.append(model(cache.get(ix),bank,pair_mask(records[first:first+len(ix)],refs,'cuda'),route=route).cpu().numpy())
                z = np.concatenate(chunks); saved[f'{arm}_{seed}_{route}'] = z
                output.append(dict(arm=arm,seed=seed,route=route,**metric(z,y)))
            output.append(dict(arm=arm,seed=seed,route='full',**metric(scores[ai,si],y)))
            output.append(dict(arm=arm,seed=seed,route='off',**metric(data['bypasses'][ai,si],y)))
            # Layer2 is frozen: all arms see identical unmodified residual/identity.
            for first in range(0,len(y),16):
                ix = np.arange(first,min(first+16,len(y))); q = cache.get(ix)
                trace = {}; x2 = model.stage2(q,trace)
                block = model.base.backbone.model.layer2[-1]
                identity = q['prefix2'].flatten(0,1)
                r = block.bn3(block.conv3(block.relu(block.bn2(block.conv2(block.relu(block.bn1(block.conv1(identity))))))))
                preact = (r+identity).reshape_as(x2)
                residual = r.reshape_as(x2)
                for p, pol in enumerate(('VV','VH')):
                    d = trace['early_2'][p]
                    update = d['update']; delta = x2[:,p]-q['raw2'][:,p]
                    pool_input = residual[:,p] if arm=='preadd' else preact[:,p] if arm=='postadd' else q['raw2'][:,p]
                    covered = (d['write_weights'].sum(1)>0)[:,None].expand_as(update)
                    for j,i in enumerate(ix):
                        active = covered[j] & (update[j].abs()>1e-8)
                        denom = int(active.sum())
                        transmissions.append(dict(arm=arm,seed=seed,id=records[i]['id'],pol=pol,
                            available=bool(q['available'][j,p]),
                            input_negative_fraction=float((pool_input[j][covered[j]]<0).float().mean()),
                            update_l2=float(update[j].norm()),effective_delta_l2=float(delta[j].norm()),
                            effective_to_proposed_l2=float(delta[j].norm()/update[j].norm().clamp_min(1e-12)),
                            zeroed_update_fraction=float(((delta[j].abs()<1e-8)&active).sum()/max(denom,1))))
            if seed == 17 and arm != 'stage':
                baseline = scores[0,0].argmax(-1); pred = scores[ai,0].argmax(-1)
                masks = dict(rescued=(baseline!=y)&(pred==y), harmed=(baseline==y)&(pred!=y),
                             CC=(baseline==y)&(pred==y), WW=(baseline!=y)&(pred!=y))
                differences = margin(scores[ai,0],y)-margin(scores[0,0],y)
                chosen = {}
                for kind,mask in masks.items():
                    candidates = np.flatnonzero(mask)
                    if len(candidates):
                        chosen[int(candidates[np.argmax(abs(differences[candidates]))])] = kind
                # Include strong disagreements with stage, not only with the no-M2 baseline.
                for kind,mask in dict(stage_rescue=(scores[1,0].argmax(-1)!=y)&(pred==y),
                                      stage_harm=(scores[1,0].argmax(-1)==y)&(pred!=y)).items():
                    candidates = np.flatnonzero(mask)
                    if len(candidates):
                        delta = margin(scores[ai,0],y)-margin(scores[1,0],y)
                        i = int(candidates[np.argmax(abs(delta[candidates]))])
                        chosen.setdefault(i,kind)
                for i,kind in chosen.items():
                    q = cache.get([i]); legal = pair_mask([records[i]],refs,'cuda')
                    z = model(q,bank,legal)
                    error = float(abs(z.cpu().numpy()[0]-scores[ai,0,i]).max())
                    assert error < 1e-4, (arm,i,error)
                    rival = int(np.where(np.arange(8)==y[i],-np.inf,scores[0,0,i]).argmax())
                    paths = {'full':entry(z,int(y[i]),rival)}
                    for p,pol in enumerate(('VV','VH')):
                        model.early[f'2_{p}'].enabled=False
                        paths[f'off_{pol}']=entry(model(q,bank,legal),int(y[i]),rival)
                        model.early[f'2_{p}'].enabled=True
                    for r,region in enumerate(('body','end','side')):
                        hooks=[mod.transforms[r].register_forward_hook(lambda module,args,value:torch.zeros_like(value)) for mod in model.early.values()]
                        try: paths[f'off_{region}']=entry(model(q,bank,legal),int(y[i]),rival)
                        finally:
                            for hook in hooks:hook.remove()
                    for mod in model.early.values():mod.enabled=False
                    paths['off_all']=entry(model(q,bank,legal),int(y[i]),rival)
                    for mod in model.early.values():mod.enabled=True
                    cases.append(dict(arm=arm,seed=seed,kind=kind,id=records[i]['id'],label=int(y[i]),rival=rival,
                        baseline_pred=int(baseline[i]), baseline_top1_margin=float(margin(scores[0,0,i:i+1],y[i:i+1])[0]),
                        paths=paths,replay_error=error))
            del model
            gc.collect();torch.cuda.empty_cache()
    np.savez(ROOT/'path_predictions.npz',**saved)
    write('path_diagnosis.json',dict(metrics=output,cases=cases,
          scope='Fixed trained weights; remove feedback from BOTH query and references. No refit or physical causal claim.'))
    write('relu_transmission.json',dict(rows=transmissions,
          scope='Algebraic transmission at layer2 only. Clipping statistics do not prove accuracy causation.'))
    print('PATH AND RELU DIAGNOSIS COMPLETE',flush=True)


if __name__=='__main__':main()
