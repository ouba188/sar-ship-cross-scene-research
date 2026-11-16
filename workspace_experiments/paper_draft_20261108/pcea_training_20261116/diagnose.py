"""Quantify constant class shifts, regional information, and learned input interaction."""
import json
from model import ROOT, Model, full, torch, np


def summary(a):
    a=np.asarray(a)
    return dict(mean=float(a.mean()),median=float(np.median(a)),p90=float(np.quantile(a,.9)),max=float(a.max()))


def main():
    inputs=full.Inputs()
    for split in ('small','validation'):inputs.prepare_geometry(split)
    selected=json.loads((ROOT/'selected_branch.json').read_text())['17']['pcea']
    net=Model(17,'pcea')
    net.load_state_dict(torch.load(ROOT/selected/'last.pt',map_location='cpu',weights_only=True,mmap=True))
    net.eval()
    ids=inputs.groups['validation'];y=inputs.y[ids]
    saved=np.load(ROOT/'withdrawal_branch_last.npz')
    original=saved['off'];actual=saved['full']
    centered=actual-original;centered-=centered.mean(1,keepdims=True)
    constant=centered.mean(0)
    approximation=original+constant
    correct=actual.argmax(1)==y;previous=original.argmax(1)==y
    report=dict(scope='seed17 branch-only LAST epoch; validation diagnostic, not a retained winner',
        constant_shift=dict(mean_centered_logits=constant.tolist(),
            fraction_squared_change=float(np.square(constant).sum()/np.square(centered).sum(1).mean()),
            matched_predictions=int((actual.argmax(1)==approximation.argmax(1)).sum()),n=len(y),
            warning='Mean estimated from observed validation score changes for descriptive diagnosis only; not a trainable control or deployable calibration.'),
        rescue=int((correct&~previous).sum()),harm=int((~correct&previous).sum()),
        product_off_prediction_changes=int((actual.argmax(1)!=saved['no_product'].argmax(1)).sum()),
        class_correct_change=[int(((y==k)&correct).sum()-((y==k)&previous).sum()) for k in range(4)])
    native=np.load(ROOT.parent/'region_dependency_20261116/native_regions.npy',mmap_mode='r')
    scores=[[],[],[]];arrays={k:[] for k in ('active','masks','token_dispersion','input_rms','update_rms','alpha','beta')}
    with torch.no_grad():
        reference=net.reference(inputs.data(inputs.groups['references'],'small'))
        for start in range(0,len(ids),12):
            part=ids[start:start+12];data=inputs.data(part,'validation')
            rows=[inputs.rows[i] for i in part]
            z,_=net(data,rows,reference,inputs.reference_rows)
            assert float(abs(z.cpu().numpy()-actual[start:start+len(part)]).max())<1e-3
            for key in arrays: arrays[key].append(net.exchange.trace[key].cpu().numpy())
            for variant,pols in enumerate(((0,),(1,),(0,1))):
                changed=dict(data);image=data['images'].clone()
                for pol in pols:
                    mask=torch.as_tensor(np.array(native[start:start+len(part),pol,0]),device='cuda')
                    x=image[:,pol,0]
                    mean=(x*mask).sum((-2,-1))/mask.sum((-2,-1)).clamp_min(1)
                    image[:,pol,0]=x+.5*mask*(mean[:,None,None]-x)
                changed['images']=image
                altered,_=net(changed,rows,reference,inputs.reference_rows)
                scores[variant].append(altered.cpu().numpy())
    arrays={k:np.concatenate(v) for k,v in arrays.items()}
    z10,z01,z11=[np.concatenate(v) for v in scores]
    rival=actual.copy();rival[np.arange(len(y)),y]=-np.inf;rival=rival.argmax(1)
    margin=lambda z:z[np.arange(len(y)),y]-z[np.arange(len(y)),rival]
    interaction=margin(z11)-margin(z10)-margin(z01)+margin(actual)
    eligible=inputs.available[ids].all(1)&native[:,:,0].any((-2,-1)).all(1)
    report['body_interaction']=dict(n=int(eligible.sum()),absolute_fixed_rival=summary(abs(interaction[eligible])),
        above_1e4=int((abs(interaction[eligible])>1e-4).sum()),
        additive_prediction_mismatch=int(((z11.argmax(1)!=(z10+z01-actual).argmax(1))&eligible).sum()))
    report['update_to_input_rms']=summary((arrays['update_rms']/np.maximum(arrays['input_rms'],1e-8))[inputs.available[ids].astype(bool)])
    m=arrays['masks'].reshape(len(ids),2,3,-1)
    cells=m.sum(-1)**2/np.maximum((m*m).sum(-1),1e-8)
    cosine=(m[:,:,0]*m[:,:,1]).sum(-1)/np.maximum(np.linalg.norm(m[:,:,0],axis=-1)*np.linalg.norm(m[:,:,1],axis=-1),1e-8)
    report['body_end_mask_cosine']=summary(cosine[inputs.available[ids].astype(bool)])
    report['regions']={name:dict(effective_cells=summary(cells[:,:,r][inputs.available[ids].astype(bool)]),
        class_token_relative_difference=summary(arrays['token_dispersion'][:,:,r][inputs.available[ids].astype(bool)]))
        for r,name in enumerate(('body','ends','side'))}
    report['alpha_between_query_std']=float(arrays['alpha'][eligible,...,0].std(axis=0).mean())
    full.write(ROOT/'DIAGNOSIS.json',report)
    np.savez_compressed(ROOT/'diagnostic_arrays.npz',ids=ids,y=y,z00=actual,z10=z10,z01=z01,z11=z11,**arrays)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
