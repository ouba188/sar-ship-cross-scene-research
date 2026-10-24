"""Exact output-Gram attribution with the established native chip renderer."""
from pathlib import Path
import importlib.util
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('distribution_analysis_run',HERE/'run.py')
run=importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)
spec=importlib.util.spec_from_file_location('existing_kernel_raw_analysis',HERE.parent/'kernel_condition_operator/analyze.py')
previous=importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)


def trace(reader,i,mb,mn):
    with torch.no_grad():
        _,raw,_,detail=reader.scores(np.array([i]),torch.tensor(mb,dtype=torch.float64),torch.tensor(mn,dtype=torch.float64),trace=True)
    d={k:v.cpu().numpy() for k,v in detail.items()}
    values=np.zeros((4,8,8,8))
    for g in range(4):
        if reader.counts[g,i]<2:
            continue
        for c in range(8):
            A=d['A'][g,0,c]
            AL=A@d['Lss'][g,c]
            for r in range(8):
                if not d['rv'][g,c,r]:
                    continue
                block=slice(8*r,8*(r+1))
                ar=A[:,block]
                m=d['ms'][g,c,r]
                linear=2*np.sum(ar*d['Lqs'][g,0,c,:,block],-1)
                energy=np.sum(ar*AL[:,block],-1)
                null=d['gamma'][g,c,r]*(np.sum(ar*ar,-1)-np.sum(ar,-1)**2/m)
                values[g,c,r]=(linear-energy+null)/d['mq'][g,0]
    error=float(np.max(abs(values.sum((-1,-2)).T-raw[0].numpy())))
    assert error<1e-10,error
    return raw[0].numpy(),values,d,error


if __name__=='__main__':
    torch.set_num_threads(2)
    previous.HERE=HERE
    previous.run=run
    previous.trace=trace
    previous.main()
