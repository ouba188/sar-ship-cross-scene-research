"""Disposable real-input gradient/interface timing check, never a reported fit."""
from pathlib import Path
import csv
import json
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
CODE=ROOT.parent/'port_prototype_model_20261121'
sys.path.insert(0,str(CODE))
from sar_port.two_stage.data import prepare_record, collate
from sar_port.two_stage.cli import fit_normalization, normalize_samples, scene_statistics, SupplementalInterventions
from sar_port.two_stage.stage1 import Stage1, Stage1Config
from sar_port.two_stage.training import stage1_optimizers, stage1_step, LossConfig, matching_step
from sar_port.two_stage.h import HModel,HConfig


def main():
    torch.set_num_threads(4)
    torch.manual_seed(9871)
    records=json.loads((ROOT.parent/'data_qualification_audit_20261122/oss2_alignment/manifest_alignment_restored_local.json').read_text(encoding='utf8'))
    with (ROOT/'q_expansion/selected_candidates_local.csv').open(encoding='utf-8-sig',newline='') as f:
        candidates=list(csv.DictReader(f))
    names=sorted({r['class_name'] for r in candidates})
    selected={c:[r for r in candidates if r['class_name']==c and r['coast_status']=='offshore'][:4] for c in names}
    donors=[selected[c][k]['sample_id'] for k in range(2) for c in names]
    queries=[selected[c][k]['sample_id'] for k in (2,3) for c in names]
    lookup={r['id']:r for r in records}; rows=[lookup[sid] for sid in donors+queries]
    samples={}
    for i,r in enumerate(rows):
        h,w=r['native_size']
        r.update(leakage_regularization_eligible=r['id'] in queries,background_donor_ids=donors,leakage_controlled=False)
        for p in ('vv','vh'):
            r[p].update(background_validated=True,background_mask=str(ROOT/f'masks_local/{r["id"]}_q.png'),
                        valid_mask=str(ROOT/f'masks_local/footprint_{h}_{w}.png'))
        samples[i]=prepare_record(r,ROOT)
    mean,std=fit_normalization(samples,list(samples));normalize_samples(samples,mean,std)
    scene_mean,scene_std=scene_statistics(samples,list(samples))
    cfg=dict(donor_ids=donors,supplemental_batches=[queries],seed=9871,batch_size=8)
    supplemental=SupplementalInterventions(samples,rows,list(samples),cfg)
    ids=list(range(8,16));batch=collate([samples[i] for i in ids],'cuda')
    supplemental.apply(batch,ids,samples)
    labels=sorted({str(r['label']) for r in rows});lm={c:i for i,c in enumerate(labels)}
    y=torch.tensor([lm[str(rows[i]['label'])] for i in ids],device='cuda')
    model=Stage1(Stage1Config(classes=4,weights_path=str(Path.home()/'.cache/torch/hub/checkpoints/convnext_tiny-983f1562.pth'))).to('cuda')
    optim=stage1_optimizers(model)
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    report=stage1_step(model,batch,y,optim,torch.full((4,),.25,device='cuda'),scene_mean,scene_std,
                       torch.zeros(8,dtype=torch.bool,device='cuda'),LossConfig(),step=200,
                       leakage_mode='empirical_regularization',regularization_views=torch.ones(8,dtype=torch.bool,device='cuda'))
    torch.cuda.synchronize();report['stage1_step_seconds']=time.perf_counter()-start
    report['stage1_peak_gpu_mb']=torch.cuda.max_memory_allocated()/1024**2
    report['finite_gate_gradient']=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for n,p in model.named_parameters() if 'gate' in n)
    assert report['controlled_samples']==0 and report['regularization_samples']==8
    assert report['scene_targets']>0 and report['discriminator']>0 and report['finite_gate_gradient']
    assert report['invariance']>0 and report['leakage']>0
    print(json.dumps(report),flush=True)
    timings=[]
    for step in (201,202):
        start=time.perf_counter()
        stage1_step(model,batch,y,optim,torch.full((4,),.25,device='cuda'),scene_mean,scene_std,
                    torch.zeros(8,dtype=torch.bool,device='cuda'),LossConfig(),step=step,
                    leakage_mode='empirical_regularization',regularization_views=torch.ones(8,dtype=torch.bool,device='cuda'))
        torch.cuda.synchronize();timings.append(time.perf_counter()-start)
    report['warm_stage1_steps_seconds']=timings
    print('warm stage1 seconds',timings,flush=True)
    del optim
    model.freeze_for_h()
    with torch.no_grad():
        query=model.encode(batch['images'],batch['valid'],batch['geometries'])
        db=collate([samples[i] for i in range(8)],'cuda')
        donor=model.encode(db['images'],db['valid'],db['geometries'])
    # Repetition is for worst-case 32-reference timing only, never a prototype bank.
    refs=donor*4;ref_labels=torch.tensor([lm[str(rows[i]['label'])] for i in range(8)]*4,device='cuda')
    h=HModel(HConfig(classes=4)).to('cuda');h.set_stage('match')
    opt=torch.optim.AdamW([p for p in h.parameters() if p.requires_grad],lr=1e-4)
    start=time.perf_counter()
    hr=matching_step(h,query,y,refs,ref_labels,['timing_only']*32,torch.ones(8,32,dtype=torch.bool,device='cuda'),opt)
    torch.cuda.synchronize();hr['batch8_reference32_seconds']=time.perf_counter()-start
    report['h_timing_only']=hr
    report['formal_training']=False
    (ROOT/'real_smoke_summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8',newline='\n')
    print(json.dumps(hr),flush=True)


if __name__=='__main__':main()
