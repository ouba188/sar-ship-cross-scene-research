"""Check real forward contracts which zero-initialized parity alone cannot test."""
from model import ROOT, RegionalExchange, torch, full


def main():
    torch.manual_seed(17)
    x=torch.randn(2,2,256,10,10)
    masks=torch.zeros(2,2,3,10,10)
    masks[:,:,0,1:4,2:8]=1
    masks[:,:,1,4:6,2:8]=1
    masks[:,:,2,6:9,2:8]=1
    responses=torch.randn(2,2,4,20,20)
    available=torch.ones(2,2)
    valid=torch.ones(2,2,1,10,10)
    report={}
    for arm in ('independent','gate','pcea'):
        model=RegionalExchange(arm)
        with torch.no_grad():
            for output in model.output: output.weight.normal_(std=.1)
            z=model(x,masks,responses,available,valid)
            delta=model.trace['delta'].clone()
            assert delta.abs().max()>0
            assert delta[...,0,:].abs().max()==0
            missing=available.clone();missing[0,1]=0
            model(x,masks,responses,missing,valid)
            assert model.trace['delta'][0].abs().max()==0
            empty=masks.clone();empty[:,:,2]=0
            model(x,empty,responses,available,valid)
            assert model.trace['delta'][...,6:9,:].abs().max()==0
            alternate=x.clone();alternate[:,1]=torch.randn_like(alternate[:,1])
            other=model(alternate,masks,responses,available,valid)
            change=float((other[:,0]-z[:,0]).abs().max())
            assert (change==0)==(arm=='independent'),(arm,change)
            report[arm]=dict(other_polarization_change_in_vv=change,missing_zero=True,
                            empty_region_zero=True,outside_zero=True)
    full.write(ROOT/'exchange_contracts.json',report)
    print(report)


if __name__=='__main__':main()
