"""Read-only native-chip integration check; random weights are not an accuracy experiment."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import json
import numpy as np
import torch
from sar_port.data import load_records,prepare_batch
from sar_port.model import ModelConfig,PortPrototypeModel
from sar_port.protocol import legal_reference_mask


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    torch.set_num_threads(2);torch.manual_seed(17)
    rows=load_records(args.manifest)
    selected=[];classes=set();vessels=set();groups=set()
    for row in rows:
        row=dict(row)
        row.setdefault('vessel_id',row.get('mmsi'))
        row.setdefault('group_id','_'.join(row.get('pass_group','').split('_')[:2]))
        if not row['vessel_id'] or not row['group_id']:
            raise ValueError('Verification requires real vessel/group identifiers')
        if row['vessel_id'] in vessels or row['group_id'] in groups:continue
        if len(selected)<2 and row['label'] in classes:continue
        if not all(row.get(p) for p in ('vv','vh')):continue
        selected.append(row);vessels.add(row['vessel_id']);groups.add(row['group_id']);classes.add(row['label'])
        if len(selected)==4:break
    if len(selected)<4:raise ValueError('Need four independent usable observations')
    refs,queries=selected[:2],selected[2:]
    rdata,qdata=prepare_batch(refs,args.device),prepare_batch(queries,args.device)
    model=PortPrototypeModel(ModelConfig(third_layer=True,gate=True)).to(args.device).eval()
    labels=torch.tensor([r['label'] for r in refs],device=args.device)
    legal=torch.tensor(legal_reference_mask(queries,refs),device=args.device)
    with torch.no_grad():
        bank=model.encode_reference(rdata)
        result=model(qdata,bank,labels,legal,details=True)
    assert torch.isfinite(result['logits']).all()
    assert (qdata['available']==1).all() and legal.all()
    assert all(torch.isfinite(t).all() for t in (bank['desc'],bank['mass'],result['attention'],result['utility'],qdata['size_condition']))
    assert result['utility'].shape==(len(queries),2,model.config.classes,8)
    report=dict(scope='engineering only; random initialization; no accuracy inference',
                query_count=len(queries),reference_count=len(refs),independent_vessels=len(vessels),
                independent_groups=len(groups),classes_in_bank=len({r['label'] for r in refs}),
                input_shape=list(qdata['images'].shape),valid_pixels=qdata['valid'].sum((-1,-2,-3)).cpu().tolist(),
                eligible_parents=qdata['eligible'].sum(-1).cpu().tolist(),
                size_prior_shape=list(qdata['size_condition'].shape),utility_shape=list(result['utility'].shape),
                gate_fusion=model.config.gate_fusion,
                finite_logits=True,mean_match_acceptance=float(result['matching']['reliability'].mean()))
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    # Private trace remains beside the user's chosen local output, never in public aggregates.
    out.with_name('native_cases_local.json').write_text(json.dumps(selected,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
