"""Extract frozen joint moments for all existing isolated historical queries."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import importlib.util
import json
import time
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import fft

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
EXT=ROOT/'external_confirmation'
spec=importlib.util.spec_from_file_location('joint_extract',ROOT/'spatial_frequency_joint_pooling/extract_joint.py')
joint=importlib.util.module_from_spec(spec);spec.loader.exec_module(joint)

def extract_rows(table, tokens, indices, extractor, snapshots, kernels):
    images,masks,energies,addresses=[],[],[],[]
    for j in indices:
        for p,pol in enumerate(('vv','vh')):
            if not tokens['valid'][j,p]:continue
            pix,fp,g,placement=joint.geo.native(table,snapshots,int(j),p)
            assert g is not None,(j,p)
            mask=joint.token.slots(g,fp)
            np.testing.assert_allclose(mask.sum((-1,-2)),tokens['mass'][j,p],rtol=1e-6,atol=1e-6)
            with Image.open(table.iloc[j]['image_'+pol]) as im:raw=np.asarray(im.convert('L'),dtype=np.float64)/255
            np.testing.assert_array_equal(joint.place(raw,placement),pix/255)
            images.append(pix.astype(np.float32)/255);masks.append(mask)
            energies.append(joint.place(joint.energy_field(raw,kernels),placement));addresses.append((j,p))
    return addresses,extractor.extract(images,masks,energies) if addresses else None

def main():
    tick=time.perf_counter();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    target=pd.read_csv(EXT/'table.csv',dtype={'mmsi':str,'instance_id':str})
    source=pd.read_csv(ROOT.parent/'regional_utility_pilot/upstream/table.csv',dtype={'mmsi':str})
    for field in ('obs_id','mmsi','pass_group'):
        assert not set(target[field])&set(source[field]),field
    tok=np.load(EXT/'tokens.npz');pr=np.load(ROOT/'projection_layer1.npz')
    np.testing.assert_array_equal(tok['instance_ids'].astype(str),target.instance_id)
    snapshots={int(s['obs_id']):s for s in json.loads((joint.geo.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    kernels=[[joint.frequency.kernel(w,a) for a in joint.frequency.ANGLES] for w in joint.frequency.WAVELENGTHS]
    extractor=joint.Extractor(pr)
    paired=pd.read_csv(ROOT/'paired_scale_readout/table.csv',dtype={'mmsi':str})
    oldtok=np.load(ROOT/'tokens.npz');ids=paired.table_index.to_numpy(int)
    small={'mass':oldtok['mass'][ids],'valid':oldtok['valid'][ids]}
    addresses,values=extract_rows(paired,small,np.arange(8),extractor,snapshots,kernels)
    cache=np.load(ROOT/'spatial_frequency_joint_pooling/features.npz');cache_error=0.
    for k,(j,p) in enumerate(addresses):
        cache_error=max(cache_error,float(abs(values[0][k].reshape(20,96)-cache['joint'][j,p]).max()))
    assert cache_error<2e-4,cache_error
    output=np.zeros((len(target),2,20,96),np.float32)
    expected=(tok['layer1']-pr['mean'])@pr['components'].T
    max_mean=0.;last=time.perf_counter()
    for first in range(0,len(target),8):
        addresses,values=extract_rows(target,tok,np.arange(first,min(first+8,len(target))),extractor,snapshots,kernels)
        for k,(j,p) in enumerate(addresses):
            active=tok['mass'][j,p]>0
            err=float(abs(values[2][k][active]-expected[j,p,active]).max(initial=0))
            assert err<2e-4,(j,p,err)
            max_mean=max(max_mean,err);output[j,p]=values[0][k].reshape(20,96)
        if first==0 or time.perf_counter()-last>25:
            print('EXTRACT',min(first+8,len(target)),'/',len(target),'seconds',round(time.perf_counter()-tick,1),'mean_replay',max_mean,flush=True);last=time.perf_counter()
    assert np.isfinite(output).all() and np.count_nonzero(output[tok['mass']<=0])==0
    np.savez_compressed(HERE/'joint_external.npz',joint=output,instance_ids=target.instance_id.to_numpy(str))
    receipt=dict(n=len(target),ships=int(target.mmsi.nunique()),passes=int(target.pass_group.nunique()),seconds=time.perf_counter()-tick,old_joint_replay=cache_error,external_mean_replay=max_mean,new_fits=0,scope='historical identity/pass-isolated coverage; not independent blind test')
    (HERE/'extraction.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print('COMPLETE',receipt,flush=True)

if __name__=='__main__':
    with fft.set_workers(2):main()
