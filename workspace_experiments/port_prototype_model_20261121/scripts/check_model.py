"""Small integration check; synthetic geometry, no research accuracy result."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import json
import tempfile
import numpy as np
from PIL import Image
import torch
from sar_port.data import prepare_batch
from sar_port.model import ModelConfig, PortPrototypeModel
from sar_port.cli import _tree_to


def synthetic_rows(directory):
    rng = np.random.default_rng(17)
    rows=[]
    for i in range(4):
        row=dict(id=str(i),vessel_id=str(i),group_id=str(i),port=str(i%2),label=i%2)
        for p in ('vv','vh'):
            pixels=rng.integers(0,80,(96,112),dtype=np.uint8)
            pixels[42:54,22:88]=rng.integers(90,240,(12,66),dtype=np.uint8)
            path=directory/f'{i}_{p}.png'
            Image.fromarray(pixels).save(path)
            row[p]=dict(image=str(path),target_index=0,detections=[dict(confidence=.9,corners=[[22,42],[88,42],[88,54],[22,54]])])
        rows.append(row)
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--output')
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(17)
    report={}
    with tempfile.TemporaryDirectory() as temp:
        rows=synthetic_rows(Path(temp))
        data=prepare_batch(rows[2:],args.device)
        refs=prepare_batch(rows[:2],args.device)
        labels=torch.tensor([0,1],device=args.device)
        legal=torch.ones(2,2,dtype=torch.bool,device=args.device)
        for name,config in (
            ('two_layer',ModelConfig(classes=2)),
            ('full',ModelConfig(classes=2,third_layer=True,gate=True,port_aggregation='hierarchical')),
        ):
            model=PortPrototypeModel(config).to(args.device).train()
            optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5)
            losses=[]
            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                reference=model.encode_reference(refs)
                result=model(data,reference,labels,legal)
                loss=torch.nn.functional.cross_entropy(result['logits'],labels)+.02*result['auxiliary'].square().mean()
                if config.gate:
                    loss=loss+.1*model.consistency_loss(data,reference,labels,legal,result['logits'],query_rows=rows[2:])
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                for group in ('backbone.resnet','selector','student','matcher','exchange')+ (('deep_context','disentangler') if config.gate else ()):
                    grads=[p.grad for n,p in model.named_parameters() if n.startswith(group) and p.grad is not None]
                    assert grads and any(g.abs().sum()>0 for g in grads),f'no gradient: {name}/{group}'
                optimizer.step()
                losses.append(float(loss.detach()))
            model.eval()
            with torch.no_grad():
                reference=model.encode_reference(refs)
                baseline=model(data,reference,labels,legal)['logits']
                path=Path(temp)/'bank.pt'
                torch.save(_tree_to(reference,'cpu',True),path)
                bank=_tree_to(torch.load(path,weights_only=True),args.device)
                frozen=model(data,bank,labels,legal)['logits']
                assert torch.equal(baseline,frozen),'frozen bank replay differs'
                # All source references excluded must remove only M3.
                no_ref=model(data,bank,labels,torch.zeros_like(legal))
                off=model(data,bank,labels,legal,m3_on=False)
                torch.testing.assert_close(no_ref['logits'],off['logits'],atol=1e-6,rtol=1e-6)
                # Query metadata / labels cannot affect the tensor-only interface.
                changed=dict(data,label=torch.tensor([1,0],device=args.device),port_index=1-data['port_index'])
                torch.testing.assert_close(model(changed,bank,labels,legal)['logits'],baseline,atol=0,rtol=0)
                missing=dict(data)
                missing['available']=data['available'].clone();missing['available'][:,1]=0
                missing['images']=data['images'].clone();missing['images'][:,1]=0
                first=model(missing,bank,labels,legal)['logits']
                missing['images'][:,1]=torch.randn_like(missing['images'][:,1])*100
                torch.testing.assert_close(model(missing,bank,labels,legal)['logits'],first,atol=0,rtol=0)
            report[name]=dict(steps=2,losses=losses,finite_gradient_norm=float(norm),frozen_bank_max_difference=float((baseline-frozen).abs().max()),missing_polarization_invariant=True)
            del result,reference,model,optimizer,loss,bank,no_ref,off
            if args.device.startswith('cuda'):torch.cuda.empty_cache()
        # Independent insertion switches use the same forward contract.
        for name,options in [('baseline',dict(m1=False,m2=False,guidance=False,pcea=False,m3=False,priors=False,effects=False,dynamic_selection=False)),
                             ('content_only',dict(priors=False,effects=False,dynamic_selection=False)),
                             ('hard_same',dict(third_layer=True,selection_mode='hard',layer_matching='same',pair_matching='aggregate',dustbin=False)),
                             ('cross',dict(third_layer=True,layer_matching='cross',axis_matching=False,class_competition=False,rejection_gate=False))]:
            model=PortPrototypeModel(ModelConfig(classes=2,**options)).to(args.device).eval()
            with torch.no_grad():
                ref=model.encode_reference(refs)
                out=model(data,ref,labels,legal)['logits']
            assert out.shape==(2,2) and torch.isfinite(out).all(),name
            report[name]=dict(finite_logits=True)
            del model,ref,out
            if args.device.startswith('cuda'):torch.cuda.empty_cache()
    if args.device.startswith('cuda'):report['peak_gib']=torch.cuda.max_memory_allocated()/2**30
    text=json.dumps(report,indent=2)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        Path(args.output).write_text(text,encoding='utf-8')

if __name__=='__main__':main()
