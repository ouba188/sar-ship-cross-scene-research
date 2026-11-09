"""Shared source-only intervention inputs; teacher decisions are model-specific."""
from pathlib import Path
import json
import numpy as np
import engine as e
from sar_m123.current import interventions

ROOT=e.ROOT/'intervention_inputs'


@e.torch.no_grad()
def prepare():
    if (ROOT/'complete.json').exists():return
    ROOT.mkdir(parents=True,exist_ok=True)
    e.task.current.setup(e.task.DEVICE)
    rows,groups=e.prev.records();a,c=e.prev.arrays()
    protocol=json.loads((e.ROOT/'protocol.json').read_text())
    net=e.Network('plain',17,protocol['pretrain'],protocol.get('classes',4));bb=net.base.backbone.model
    groups_out=[];total=0
    for idx in groups['source_adapt']:
        items,meta=interventions.variants(rows[idx])
        groups_out.append(dict(index=int(idx),id=rows[idx]['id'],start=total,metadata=meta))
        total+=len(items)-1
    xout=np.lib.format.open_memmap(ROOT/'early_x.npy',mode='w+',dtype=np.float32,shape=(total,128,20,20))
    iout=np.lib.format.open_memmap(ROOT/'identity.npy',mode='w+',dtype=np.float32,shape=(total,512,20,20))
    errors=[]
    for n,group in enumerate(groups_out):
        idx=group['index'];items,meta=interventions.variants(rows[idx])
        assert meta==group['metadata']
        for first in range(0,len(items),8):
            batch=items[first:first+8]
            images=e.torch.tensor(np.stack([v['images'] for v in batch]),device=e.task.DEVICE)
            z=net.base.prefix(images,stop=1).flatten(0,1)
            for block in list(bb.layer2.children())[:3]:z=block(z)
            identity=z.reshape(len(batch),2,512,20,20)
            block=bb.layer2[3]
            x=block.relu(block.bn1(block.conv1(z)))
            x=block.relu(block.bn2(block.conv2(x))).reshape(len(batch),2,128,20,20)
            for j in range(len(batch)):
                k=first+j
                if k==0:
                    err=max(float(abs(x[j].cpu().numpy()-c['early_x'][idx]).max()),
                            float(abs(identity[j].cpu().numpy()-c['early_identity'][idx]).max()))
                    assert err<1e-4,(idx,err);errors.append(err)
                else:
                    p=('vv','vh').index(meta[k]['pol']);out=group['start']+k-1
                    xout[out]=x[j,p].cpu().numpy();iout[out]=identity[j,p].cpu().numpy()
        if (n+1)%30==0:print('INTERVENTION_INPUTS',n+1,len(groups_out),flush=True)
    xout.flush();iout.flush()
    e.task.write(ROOT/'complete.json',dict(count=total,groups=groups_out,original_error=max(errors),
        dtype='float32',scope='source_adapt only; frozen ImageNet prefix; no teacher labels or decisions cached'))
    print('INTERVENTION_INPUTS_DONE',total,max(errors),flush=True)


@e.torch.no_grad()
def select(net,a,c,rows,ids,rid):
    receipt=json.loads((ROOT/'complete.json').read_text())
    x=np.load(ROOT/'early_x.npy',mmap_mode='r');identity=np.load(ROOT/'identity.npy',mmap_mode='r')
    result=[]
    for group in receipt['groups']:
        idx=group['index'];assert idx in ids
        meta=group['metadata'];scores=[]
        for first in range(0,len(meta),8):
            entries=meta[first:first+8];indices=np.repeat(idx,len(entries))
            changed={k:e.task.tensor(c[k],indices) for k in ('early_x','early_identity','local_slots')}
            for j,item in enumerate(entries):
                k=first+j
                if k:
                    p=('vv','vh').index(item['pol']);pos=group['start']+k-1
                    changed['early_x'][j,p]=e.torch.from_numpy(np.array(x[pos],copy=True)).to(e.task.DEVICE)
                    changed['early_identity'][j,p]=e.torch.from_numpy(np.array(identity[pos],copy=True)).to(e.task.DEVICE)
            local_a={key:e.task.tensor(a[key],indices) for key in ('valid','available','slots')}
            # No M1 or M3 is active yet. Geometry/availability stay unchanged.
            z=net(local_a,changed,[rows[idx]]*len(entries),np.arange(len(entries)),[])[0]
            scores.append(z.cpu().numpy())
        scores=np.concatenate(scores);y=rows[idx]['label']
        choice=interventions.select_teacher(scores,meta,y)
        clean=e.torch.tensor(scores[0]);altered=e.torch.tensor(scores[choice['chosen_index']])
        def margin(z):
            other=z.clone();other[y]=-e.torch.inf
            return z[y]-other.max()
        useful=choice['useful'] and bool((altered.log_softmax(0)[y]>clean.log_softmax(0)[y]) & (margin(altered)>margin(clean)))
        result.append(dict(index=idx,id=rows[idx]['id'],label=y,useful=useful,
            original=clean.tolist(),altered=altered.tolist(),choice=choice,selected_intervention=meta[choice['chosen_index']]))
    return result


if __name__=='__main__':prepare()
