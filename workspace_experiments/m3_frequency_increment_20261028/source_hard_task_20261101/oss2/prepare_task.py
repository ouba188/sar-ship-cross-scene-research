"""Freeze source-only cache before choosing any new trainable checkpoint."""
import time
import numpy as np
import torch
from model_task import *
from geometry import build_masks,pool_multiscale


def main():
    current.setup(DEVICE)
    rows,train,hold,rid=records();dest=ROOT/'cache';dest.mkdir(exist_ok=True)
    if (dest/'complete.json').exists():print('CACHE_EXISTS',flush=True);return
    net=base_model()
    keys=(set(data.SHAPES)-{'images'})|{'prefix'}
    old={k:np.load(PUBLIC/'cache'/f'{k}.npy',mmap_mode='r') for k in keys}
    def take(rr):
        ix=[r['indices']['table_index'] for r in rr]
        result={k:tensor(a,ix) for k,a in old.items()}
        for k in data.FREQUENCY_KEYS:result[k].zero_()
        return result
    out={};tick=time.perf_counter()
    with torch.no_grad():
        dummy=take([rows[rid[0]]])
        for first in range(0,len(rows),8):
            part=rows[first:first+8];b=take(part);size=len(part)
            v=b['valid']*b['available'][...,None,None,None]
            x2,x,_,raw3,_,raw2,_=net.stage3_paths(b['prefix'],v,b['available'],b['field2'],b['field3'],
                dummy,dummy['valid'],b['slot_masks'],b['frequency'],{},[],b['local_slot_masks'],b['local_frequency'])
            k3=[];k2=[];mass=[];low=[]
            for p,op in enumerate(net.m3[0]):
                ks,m=op.descriptors(raw3[:,p],raw2[:,p],b['slot_masks'][:,p],b['local_slot_masks'][:,p],v[:,p],
                    b['frequency'][:,p],b['local_frequency'][:,p])
                k3.append(ks[0]);k2.append(ks[1]);mass.append(m)
                reg,en=net.region_integrals(b['local_slot_masks'][:,p],b['local_frequency'][:,p])
                low.append(net.m2[p](x2[:,p],reg,v[:,p],en)['pooled'])
            imgs=np.zeros((size,2,1,160,160),np.float32);geo=[]
            for n,row in enumerate(part):
                views=data.prepare(row);gs=[]
                for p,view in enumerate(views):
                    present=view is not None and view['valid_target']
                    if present:imgs[n,p,0]=view['pixels'].astype(np.float32)/255
                    gs.append(build_masks(view['geometry'] if present else None,
                        view['footprint'] if present else np.zeros((160,160)),'coarse_fixed'))
                geo.append(gs)
            f1=net.prefix(torch.tensor(imgs,device=DEVICE),stop=1)
            read=torch.tensor(np.stack([[g['read'] for g in gs] for gs in geo]),device=DEVICE)
            desc=pool_multiscale(f1.flatten(0,1),b['prefix'].flatten(0,1),read.flatten(0,1),'coarse_fixed').reshape(size,2,20,768)
            vals=dict(pre_m3=x,key3=torch.stack(k3,1),key2=torch.stack(k2,1),original_mass=torch.stack(mass,1),
                low=torch.stack(low,1),slots=b['slot_masks'],valid=v,available=b['available'],desc=desc,
                mass=np.stack([[g['mass'] for g in gs] for gs in geo]),write=np.stack([[g['write'] for g in gs] for gs in geo]))
            for key,value in vals.items():
                arr=value.cpu().numpy() if isinstance(value,torch.Tensor) else value
                if key not in out:out[key]=np.lib.format.open_memmap(dest/f'{key}.npy',mode='w+',dtype=arr.dtype,shape=(len(rows),)+arr.shape[1:])
                out[key][first:first+size]=arr
            if first%80==0:
                for a in out.values():a.flush()
                print('CACHE',first+size,len(rows),round(time.perf_counter()-tick,1),flush=True)
        for a in out.values():a.flush()
        # Compare cached wrapper with the actual complete spatial public network.
        ids=np.r_[train[:2],hold[:2]];bank=take([rows[i] for i in rid]);b=take([rows[i] for i in ids])
        expected=net.forward_from(**b,reference=bank,pair_valid=legal(rows,ids,rid)).cpu().numpy()
        del net
        model=Model('original');actual=predict(model,out,rows,ids,rid)['score']
        parity=float(np.abs(actual-expected).max());assert parity<1e-4,parity
        predictions=predict(model,out,rows,np.arange(len(rows)),rid)
        np.savez_compressed(ROOT/'baseline.npz',**predictions)
        y=np.array([r['label'] for r in rows]);z=predictions['score']
        margin=z[np.arange(len(rows)),y]-np.where(np.arange(4)[None]==y[:,None],-np.inf,z).max(1)
        write(ROOT/'source_difficulty.json',dict(train=train.tolist(),validation=hold.tolist(),references=rid.tolist(),
            hard=train[margin[train]<1].tolist(),wrong=train[margin[train]<0].tolist(),
            source_metrics=metrics(y[train],z[train]),baseline_validation=metrics(y[hold],z[hold]),
            margin_train=margin[train].tolist()))
        print('BASELINE',metrics(y[train],z[train]),metrics(y[hold],z[hold]),'hard',int((margin[train]<1).sum()),flush=True)
    write(dest/'complete.json',dict(n=len(rows),train=len(train),validation=len(hold),frequency=False,parity_error=parity,
        evaluation_manifest_read=False,seconds=time.perf_counter()-tick,checkpoint=str(PUBLIC/'training/full_s17/adapt/model.pt')))


if __name__=='__main__':main()
