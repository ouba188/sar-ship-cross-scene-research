"""Frozen pathway dependence for new depth candidates; query and references both change."""
import gc
from run import ROOT,ARMS,SEEDS,Cache,construct,setup,pair_mask,write,np,torch,checkpoint_path,load_inputs


def margin(z,y):
    other=z.copy();other[np.arange(len(y)),y]=-np.inf
    return z[np.arange(len(y)),y]-other.max(1)


def entry(score,label,rival):
    z=score.detach().cpu().numpy()[0]
    return dict(pred=int(z.argmax()),top1_margin=float(margin(z[None],np.array([label]))[0]),
        fixed_margin=float(z[label]-z[rival]))


@torch.inference_mode()
def main():
    setup('cuda');state,records=load_inputs()
    refs,rows=records['refs'],records['dev']
    cache=Cache('dev');bank=Cache('refs').get(np.arange(len(refs)))
    saved=np.load(ROOT/'predictions.npz');scores=saved['scores'];y=saved['labels']
    cases=[]
    for arm in ('block1','block2'):
        ai=list(ARMS).index(arm);seed=17;si=SEEDS.index(seed)
        base=scores[0,si].argmax(-1);pred=scores[ai,si].argmax(-1)
        masks=dict(rescued=(base!=y)&(pred==y),harmed=(base==y)&(pred!=y),
            CC=(base==y)&(pred==y),WW=(base!=y)&(pred!=y))
        dif=margin(scores[ai,si],y)-margin(scores[0,si],y)
        chosen={}
        for kind,mask in masks.items():
            ix=np.flatnonzero(mask)
            if len(ix):chosen[int(ix[np.argmax(abs(dif[ix]))])]=kind
        last=scores[list(ARMS).index('block3'),si]
        for kind,mask in dict(last_rescue=(last.argmax(-1)!=y)&(pred==y),
                last_harm=(last.argmax(-1)==y)&(pred!=y)).items():
            ix=np.flatnonzero(mask)
            if len(ix):
                delta=margin(scores[ai,si],y)-margin(last,y)
                chosen.setdefault(int(ix[np.argmax(abs(delta[ix]))]),kind)
        model=construct(state,ARMS[arm],seed)
        model.load_state_dict(torch.load(checkpoint_path(arm,seed),weights_only=True,map_location='cpu')['state'],strict=False)
        for i,kind in chosen.items():
            q=cache.get([i]);legal=pair_mask([rows[i]],refs,'cuda')
            z=model(q,bank,legal)
            error=float(abs(z.cpu().numpy()[0]-scores[ai,si,i]).max())
            assert error<1e-4,(arm,i,error)
            rival=int(np.where(np.arange(8)==y[i],-np.inf,scores[0,si,i]).argmax())
            paths=dict(full=entry(z,int(y[i]),rival))
            for route in ('backbone_only','readout_only'):
                paths[route]=entry(model(q,bank,legal,route=route),int(y[i]),rival)
            for p,pol in enumerate(('VV','VH')):
                model.early[f'2_{p}'].enabled=False
                paths[f'off_{pol}']=entry(model(q,bank,legal),int(y[i]),rival)
                model.early[f'2_{p}'].enabled=True
            for r,region in enumerate(('body','end','side')):
                hooks=[module.transforms[r].register_forward_hook(lambda m,args,value:torch.zeros_like(value)) for module in model.early.values()]
                try:paths[f'off_{region}']=entry(model(q,bank,legal),int(y[i]),rival)
                finally:
                    for hook in hooks:hook.remove()
            for module in model.early.values():module.enabled=False
            paths['off_all']=entry(model(q,bank,legal),int(y[i]),rival)
            for module in model.early.values():module.enabled=True
            cases.append(dict(arm=arm,seed=seed,kind=kind,id=rows[i]['id'],label=int(y[i]),rival=rival,
                baseline_pred=int(base[i]),baseline_top1_margin=float(margin(scores[0,si,i:i+1],y[i:i+1])[0]),
                block3_pred=int(last[i].argmax()),paths=paths,replay_error=error))
        del model
        gc.collect();torch.cuda.empty_cache()
    write('path_diagnosis.json',dict(cases=cases,scope='Frozen weights; feedback removed for BOTH query and reference. Posthoc pathway dependence, not physical or pixel attribution.'))
    print('PATH DIAGNOSIS COMPLETE',flush=True)


if __name__=='__main__':main()
