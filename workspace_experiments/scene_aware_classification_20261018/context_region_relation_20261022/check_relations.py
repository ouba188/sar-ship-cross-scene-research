"""Check residual fallback and unordered-neighbor contracts before training."""
import json
import torch
from run_relations import HERE, ARMS, Relation, load_data

torch.set_num_threads(2)
torch.manual_seed(101)
q=torch.randn(3,32)
n=torch.randn(3,4,64)
g=torch.randn(3,4,7)
stats=torch.randn(3,15)
m=torch.tensor([[False]*4,[True,False,False,False],[True]*4])
perm=torch.tensor([2,0,3,1])
rows=[]
for arm in ARMS:
    model=Relation(arm).eval()
    assert torch.count_nonzero(model(q,n,g,stats,m))==0
    with torch.no_grad():
        model.head.weight.normal_();model.head.bias.normal_()
    a=model(q,n,g,stats,m)
    b=model(q,n[:,perm],g[:,perm],stats,m[:,perm])
    assert torch.count_nonzero(a[0])==0
    error=float((a-b).abs().max().detach())
    assert error<1e-5,(arm,error)
    rows.append(dict(arm=arm,zero_initial_residual=True,no_neighbor_fallback=True,permutation_max_error=error))
t,f,r=load_data()
assert not r['fg_empty'].any() and not r['bg_empty'].any()
(HERE/'structural_checks.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
print('Passed:',len(rows),'arms; nonempty readouts; frozen target IDs and edges.')
