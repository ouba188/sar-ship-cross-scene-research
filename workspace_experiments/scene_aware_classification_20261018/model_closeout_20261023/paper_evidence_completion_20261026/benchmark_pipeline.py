"""Warm single-chip latency of the actual exported eight-class pipeline."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
PACKAGE=Path('E:/临时会话/sar_m123_standalone')
sys.path.insert(0,str(PACKAGE))
from sar_m123.pipeline import ShipClassifier
from sar_m123.input import prepare
from sar_m123.baseline import extract as baseline_extract


def main():
    device='cuda' if torch.cuda.is_available() else 'cpu'
    def sync():
        if device=='cuda':torch.cuda.synchronize()
    records=json.loads((PACKAGE/'validation_local/requests.json').read_text(encoding='utf-8'))
    out=HERE/'compute_cost'
    out.mkdir(exist_ok=True)
    tick=time.perf_counter()
    model=ShipClassifier(PACKAGE/'assets',device=device)
    load_seconds=time.perf_counter()-tick
    directory=PACKAGE/'validation_local'
    # Load per-seed heads and warm kernels without reusing query features in timing.
    model.predict(records[0],directory,seeds=(17,))
    model.predict(records[0],directory,seeds=(17,))
    sync()
    rows=[]
    with torch.inference_mode():
        for repeat in range(3):
            for record in records:
                if device=='cuda':torch.cuda.reset_peak_memory_stats()
                sync();t0=time.perf_counter()
                prepared=prepare(record,directory)
                sync();t1=time.perf_counter()
                maps=model.backbone(prepared)
                sync();t2=time.perf_counter()
                baseline=baseline_extract(prepared,maps,model.backbone)
                sync();t3=time.perf_counter()
                base,_=model.baselines[17](baseline)
                sync();t4=time.perf_counter()
                target=[v if v is not None and v['valid_target'] else None for v in prepared]
                target_maps=[m if v is not None else None for v,m in zip(target,maps)]
                summaries=model.summary_reader.summarize(target,target_maps)
                sync();t5=time.perf_counter()
                s12,_=model.m12[17](base,summaries)
                sync();t6=time.perf_counter()
                local=model.m3.extract(target,target_maps)
                sync();t7=time.perf_counter()
                score,_=model.m3(local,s12,17)
                sync();t8=time.perf_counter()
                row=dict(id=record['id'],repeat=repeat,available=sum(v is not None for v in target),
                    prepare_ms=1000*(t1-t0),shared_backbone_ms=1000*(t2-t1),
                    baseline_features_ms=1000*(t3-t2),baseline_head_ms=1000*(t4-t3),
                    m12_summaries_ms=1000*(t5-t4),m12_heads_ms=1000*(t6-t5),
                    m3_local_frequency_ms=1000*(t7-t6),m3_match_fuse_ms=1000*(t8-t7),
                    total_ms=1000*(t8-t0),baseline_path_ms=1000*(t4-t0),
                    predicted=int(np.argmax(score)),
                    gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if device=='cuda' else None)
                rows.append(row)
                del maps,target_maps
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'timings.csv',index=False)
    frame.groupby('available').agg({k:['median','mean','std'] for k in frame if k.endswith('_ms') or k.endswith('_mib')}).to_csv(out/'summary.csv')
    active=sum(p.numel() for part in (model.backbone.model.conv1,model.backbone.model.bn1,
        model.backbone.model.layer1,model.backbone.model.layer2,model.backbone.model.layer3) for p in part.parameters())
    params={'backbone_resident':sum(p.numel() for p in model.backbone.parameters()),
        'backbone_executed_through_layer3':active,
        'M1':sum(p.numel() for p in model.m12[17].heads['attention'].parameters()),
        'M2':sum(p.numel() for p in model.m12[17].heads['full'].parameters())}
    m3=model.m3._load(17)
    params['M3_modules']={k:sum(p.numel() for p in v.parameters()) for k,v in m3.items() if isinstance(v,torch.nn.Module)}
    receipt=dict(device=device,gpu=torch.cuda.get_device_name() if device=='cuda' else None,
        pytorch=torch.__version__,model_load_seconds=load_seconds,classes=8,reference_bank=64,
        native_chips=len(records),repeats=3,seed=17,batch=1,parameters=params,
        bank_arrays_bytes=sum(x.nbytes for x in model.m3.bank.values()),
        scope='actual standalone8class, not public4class; warm filesystem/kernel; image IO included; detection/model loading excluded',
        baseline_path='includes shared feature forward through layer3; not an independently optimized stem-only baseline',
        inference='features freshly extracted every timed chip, no query-cache reuse')
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False),encoding='utf-8')
    print(frame.groupby('available')[['total_ms','baseline_path_ms','m3_match_fuse_ms']].mean().to_string(),flush=True)


if __name__=='__main__':main()
