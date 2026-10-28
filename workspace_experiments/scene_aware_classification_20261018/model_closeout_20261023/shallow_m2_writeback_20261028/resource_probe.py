import time
import torch
from run import RELEASE, ROOT, Cache, construct, read_manifest, pair_mask, setup, write

setup('cuda')
state=torch.load(RELEASE/'weights/teacher.pt',map_location='cpu',weights_only=True)['state']
model=construct(state,(1,2),17)
records=read_manifest(RELEASE/'manifests/source_adapt.json')
refs=read_manifest(RELEASE/'manifests/references.json')
bank=Cache('refs').get(list(range(16)))
query=Cache('source').get(list(range(16)))
legal=pair_mask(records[:16],refs,'cuda')
labels=torch.tensor([r['label'] for r in records[:16]],device='cuda')
times=[]
torch.cuda.reset_peak_memory_stats()
for i in range(3):
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize();start=time.perf_counter()
    torch.nn.functional.cross_entropy(model(query,bank,legal),labels).backward()
    torch.cuda.synchronize();times.append(time.perf_counter()-start)
result=dict(batch=16,arm='both',forward_backward_seconds=times,peak_cuda_mb=torch.cuda.max_memory_allocated()/1024**2)
write('resource_probe.json',result)
print(result,flush=True)
