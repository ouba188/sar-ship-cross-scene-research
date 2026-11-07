"""Check four-class integration and preserve the eight-class default interface."""
from torch.nn import functional as F
from model_task import *


def main():
    current.setup(DEVICE);rows,train,hold,rid=records();a=arrays()
    d=json.loads((ROOT/'source_difficulty.json').read_text());ids=np.array(d['hard'][:4]);y=torch.tensor([rows[i]['label'] for i in ids],device=DEVICE)
    default=MaskedCostLocalMatch(17);explicit=MaskedCostLocalMatch(17,classes=8)
    assert all(torch.equal(v,explicit.state_dict()[k]) for k,v in default.state_dict().items())
    del default,explicit
    receipts=[];baseline=None;candidate=None
    for arm in ('none','aggregate','competitive','original'):
        model=Model(arm);z,ratio=model(a,rows,ids,rid)
        assert z.shape==(len(ids),4) and torch.isfinite(z).all()
        if arm=='none':baseline=z.detach().clone()
        if arm in ('aggregate','competitive'):
            assert torch.equal(z,baseline) and float(ratio.max())==0
            if candidate is None:candidate={k:v.clone() for k,v in model.matcher.state_dict().items()}
            else:assert all(torch.equal(v,candidate[k]) for k,v in model.matcher.state_dict().items())
        F.cross_entropy(z,y).backward()
        gradients={n:float(p.grad.norm()) for n,p in model.named_parameters() if p.grad is not None}
        assert gradients['base.head.weight']>0
        assert not any(n.startswith('base.backbone.') for n in gradients)
        if arm!='none':assert any(v>0 for n,v in gradients.items() if n.startswith('matcher.'))
        assert all(np.isfinite(list(gradients.values())))
        receipts.append(dict(arm=arm,gradients=gradients,output_shape=list(z.shape)))
        del model,z
    write(ROOT/'integration_checks.json',dict(source_ids=[rows[i]['id'] for i in ids],eight_class_default_unchanged=True,
        four_class_output=True,candidate_same_init=True,zero_writeback_equal_none=True,checks=receipts,fits=0))
    print('CHECKS_PASS',flush=True)


if __name__=='__main__':main()
