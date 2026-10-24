"""Shared native-image Gaussian views and frozen source projection."""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '2'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json
import time
import importlib.util
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
spec = importlib.util.spec_from_file_location('pixel_probe', ROOT/'image_frequency_probe/run.py')
pixel = importlib.util.module_from_spec(spec); spec.loader.exec_module(pixel)
geo = pixel.geo


class SharedExtractor(pixel.Extractor):
    @torch.inference_mode()
    def unnormalized(self, images, masks):
        output = []
        for first in range(0, len(images), 32):
            ims = images[first:first+32]; ms = masks[first:first+32]
            xx = torch.tensor(np.stack(ims)[:, None], device='cuda')
            raw = np.zeros((len(ims), 20, 256), np.float32)
            for k in range(4):
                v = xx.rot90(k, (-2,-1)).expand(-1,3,-1,-1)
                z = self.model.relu(self.model.bn1(self.model.conv1((v-self.mean)/self.std)))
                z = self.model.layer1(self.model.maxpool(z))
                w = np.stack([pixel.cell_mass(np.rot90(m,k,(-2,-1)),4) for m in ms]).reshape(len(ms),20,-1)
                w = w/np.maximum(w.sum(-1,keepdims=True),1e-20)
                desc = torch.einsum('btu,bdu->btd',torch.tensor(w,device='cuda',dtype=torch.float32),z.flatten(2))
                raw += desc.cpu().numpy()/4
            value = ((raw-self.projection['mean'])@self.projection['components'].T).astype(np.float32)
            value[np.stack(ms).sum((-1,-2)) <= 0] = 0
            output.append(value)
        return np.concatenate(output)


def main():
    tick=time.perf_counter(); torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    table=pd.read_csv(pixel.UP/'table.csv',dtype={'mmsi':str,'instance_id':str})
    pr=np.load(ROOT/'projection_layer1.npz'); bank=pr['bank']
    queries=np.flatnonzero(table.role.isin(['utility_train','selection','evaluation']))
    ids=np.unique(np.r_[bank.ravel(),queries]); assert len(ids)==1805
    fit=np.load(pixel.UP/'indices.npz')['fit_train']; assert set(bank.ravel())<=set(fit)
    for a,b in [('utility_train','selection'),('utility_train','evaluation'),('selection','evaluation')]:
        for field in ('mmsi','pass_group'):
            assert not set(table[table.role.eq(a)][field])&set(table[table.role.eq(b)][field])
    assert not set(table.iloc[bank.ravel()].mmsi)&set(table.iloc[queries].mmsi)
    original=np.load(ROOT/'tokens.npz'); np.testing.assert_array_equal(original['instance_ids'].astype(str),table.instance_id)
    raw=original['layer1'][ids]; mass=original['mass'][ids]; valid=original['valid'][ids]
    y=np.zeros((len(ids),2,4,20,32),np.float32)
    y[:,:,0]=(raw-pr['mean'])@pr['components'].T
    y[:,:,0][mass<=0]=0
    del raw
    table.iloc[ids].assign(table_index=ids).to_csv(HERE/'table.csv',index=False)
    snapshots={int(s['obs_id']):s for s in json.loads((geo.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    extractor=SharedExtractor(pr); provenance=[]; errors=[]
    for first in range(0,len(ids),16):
        images=[]; masks=[]; addresses=[]
        for j in range(first,min(first+16,len(ids))):
            i=ids[j]
            for p in range(2):
                if not valid[j,p]: continue
                pix,fp,g,place=geo.native(table,snapshots,int(i),p)
                mask=pixel.slots(g,fp)
                with Image.open(table.iloc[i]['image_'+('vv','vh')[p]]) as im:
                    raw_image=np.array(im.convert('L'),dtype=np.float32)/255
                if first==0:
                    images.append(pix.astype(np.float32)/255); masks.append(mask); addresses.append((j,p,0))
                for s,sigma in enumerate((1,4,12),1):
                    view=pixel.place(gaussian_filter(raw_image,sigma,mode='reflect'),place)
                    assert np.max(abs(view[fp==0]),initial=0)==0
                    images.append(view); masks.append(mask); addresses.append((j,p,s))
                provenance.append(dict(obs_id=int(table.iloc[i].obs_id),pol=p,path=table.iloc[i]['image_'+('vv','vh')[p]],placement=place))
        if images:
            result=extractor.unnormalized(images,masks)
            for value,(j,p,s) in zip(result,addresses):
                if s==0:
                    errors.append(float(abs(value-y[j,p,0]).max()))
                else:
                    y[j,p,s]=value
        if first%128==0:
            print('VIEWS',min(first+16,len(ids)),'/',len(ids),round(time.perf_counter()-tick,1),flush=True)
    assert max(errors)<2e-4,errors
    np.savez_compressed(HERE/'views.npz',y=y,mass=mass,valid=valid,ids=ids,bank=bank)
    (HERE/'image_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    receipt=dict(n=len(ids),roles=table.iloc[queries].role.value_counts().to_dict(),reference_bank=64,
        valid_views=int(valid.sum()),raw_replay=max(errors),seconds=time.perf_counter()-tick,scales=[0,1,4,12],
        spatial_units='native PNG pixels, not matched physical meters',new_backbone_fits=0)
    (HERE/'extraction.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(receipt,flush=True)


if __name__=='__main__': main()
