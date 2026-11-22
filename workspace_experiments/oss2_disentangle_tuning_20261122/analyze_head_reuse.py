"""Validation-only readout substitution for a fixed, already-trained H.

Reuses cached H evidence E, comparing its original h0 logits with the original
Stage1 disentangle classification head. This is not a retrained H candidate.
No Torch import, model/checkpoint loading, images, or test predictions.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
SEEDS = (17,29,43)
GAMMAS = (0.,.25,.5,1.,2.,4.)
TEMPERATURE = .2
METRICS = ('balanced_accuracy','accuracy','macro_f1')
STAGES = ('baseline_stage1','baseline_base','disentangle_stage1','disentangle_base','full_h')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            result.update(block)
    return result.hexdigest()


def metrics(y, pred, classes, np):
    matrix = np.zeros((len(classes),len(classes)),dtype=np.int64)
    np.add.at(matrix,(y,pred),1)
    support,predicted = matrix.sum(1),matrix.sum(0)
    correct = matrix.diagonal()
    recall = np.divide(correct,support,out=np.zeros(len(classes)),where=support>0)
    f1 = np.divide(2*correct,support+predicted,out=np.zeros(len(classes)),where=support+predicted>0)
    return dict(count=int(len(y)),accuracy=float(correct.sum()/len(y)),
        balanced_accuracy=float(recall[support>0].mean()),macro_f1=float(f1.mean()),
        confusion_matrix=matrix.tolist(),per_class={str(c):dict(count=int(support[i]),recall=float(recall[i]),f1=float(f1[i]))
                                                  for i,c in enumerate(classes)})


def metric_replay(actual, expected, name):
    if any(actual[k]!=expected[k] for k in ('count','confusion_matrix')):
        raise ValueError(name+': count/confusion matrix replay differs')
    if any(abs(actual[k]-expected[k])>1e-12 for k in METRICS):
        raise ValueError(name+': floating metric replay differs')


def read_natural(path, np):
    with np.load(path,allow_pickle=False) as data:
        # Do not decompress the much larger Fc/Fs arrays.
        result={key:data[key] for key in ('ids','split','label','class_order','stage1_logits')}
        result['metadata']=json.loads(str(data['metadata_json'].item()))
    return result


def align_heads(natural, query_ids, targets, classes, np):
    ids=[str(x) for x in natural['ids']]
    splits=[str(x) for x in natural['split']]
    labels=[str(x) for x in natural['label']]
    logits=natural['stage1_logits']
    if len(ids)!=945 or len(set(ids))!=945 or splits.count('train')!=718 or splits.count('validation')!=227:
        raise ValueError('Natural cache must contain unique 718 train + 227 validation observations')
    if len(labels)!=len(ids) or len(splits)!=len(ids) or logits.shape!=(len(ids),len(classes)):
        raise ValueError('Natural logits/metadata shape mismatch')
    if list(natural['class_order'])!=classes or natural['metadata']['class_order']!=classes:
        raise ValueError('Natural and H cache class order differs')
    if logits.dtype!=np.float32 or not np.isfinite(logits).all():
        raise ValueError('Expected finite float32 Stage1 logits')
    lookup={sid:i for i,sid in enumerate(ids)}
    validation={sid for sid,split in zip(ids,splits) if split=='validation'}
    if len(query_ids)!=227 or len(set(query_ids))!=227 or set(query_ids)!=validation:
        raise ValueError('H query identities differ from the Stage1 validation split')
    order=np.asarray([lookup[sid] for sid in query_ids],dtype=np.int64)
    if any(labels[i]!=classes[int(y)] for i,y in zip(order,targets)):
        raise ValueError('Aligned query labels differ between Stage1 and H caches')
    return logits[order]


def scan(z0, evidence, targets, classes, baseline_pred, np):
    original_pred=z0.argmax(1); original_correct=original_pred==targets
    baseline_correct=baseline_pred==targets
    baseline_metrics=metrics(targets,baseline_pred,classes,np)
    rows=[]
    for gamma in GAMMAS:
        pred=(z0+np.float32(gamma/TEMPERATURE)*evidence).argmax(1)
        correct=pred==targets
        rescued=(~original_correct)&correct; harmed=original_correct&(~correct)
        result=metrics(targets,pred,classes,np)
        result.update(gamma=gamma,rescued_vs_own_gamma0=int(rescued.sum()),harmed_vs_own_gamma0=int(harmed.sum()),
            rescued_vs_stage1_baseline=int(((~baseline_correct)&correct).sum()),
            harmed_vs_stage1_baseline=int((baseline_correct&(~correct)).sum()),
            delta_vs_stage1_baseline_pp={key:100*(result[key]-baseline_metrics[key]) for key in METRICS},
            rescue_harm_by_class={str(c):dict(rescued=int((rescued&(targets==i)).sum()),
                                            harmed=int((harmed&(targets==i)).sum())) for i,c in enumerate(classes)})
        rows.append(result)
    return rows


def readiness(experiment, branch_root, fusion_root):
    paths=[]
    for seed in SEEDS:
        paths += [branch_root/f'seed{seed}'/'branch_diagnostics_features_local'/f'{arm}_natural_features_local.npz'
                  for arm in ('baseline','disentangle')]
        paths += [fusion_root/f'final_seed{seed}'/name for name in
                  ('validation_logits_evidence_local.npz','aggregate_validation_fusion_scan.json')]
    paths += [experiment/'aggregate_training_integrity.json',experiment/'ledger_local.json',experiment/'state_local.json']
    if not all(path.is_file() for path in paths):
        raise ValueError('All three seeds need completed natural-head caches, final H caches, and training integrity first')
    if (experiment/'.run_comparison.lock').exists():
        raise ValueError('Original comparison is still running')
    integrity=read_json(experiment/'aggregate_training_integrity.json')
    if integrity.get('status')!='passed' or {r['seed'] for r in integrity['seeds']}!=set(SEEDS):
        raise ValueError('Completed three-seed training integrity is required')
    required={'encoder_stage1_to_base_exact','encoder_base_to_h_exact','h0_base_to_h_exact','checkpoint_stage_tags'}
    for result in integrity['seeds']:
        checks={item['name']:item for item in result['checks']}
        if not result['passed'] or any(not checks.get(name,{}).get('passed',False) for name in required):
            raise ValueError('Frozen encoder/h0 stage consistency has not passed')
    ledger=read_json(experiment/'ledger_local.json'); state=read_json(experiment/'state_local.json')
    for seed in SEEDS:
        for stage in STAGES:
            if state['steps'].get(f'seed{seed}/{stage}',{}).get('status')!='succeeded':
                raise ValueError('All original training stages must have succeeded')
    inputs=ledger['inputs']; manifest=experiment/'experiment_manifest_local.json'
    if digest(manifest)!=inputs['manifest']['sha256']:
        raise ValueError('Experiment manifest no longer matches its pin')
    for relative,expected in inputs['code_sha256'].items():
        if digest(Path(inputs['code_root'])/relative)!=expected:
            raise ValueError('Model source no longer matches its pin')
    records=read_json(manifest); records=records if isinstance(records,list) else records['records']
    records=[r for r in records if r.get('development_split') in ('train','val','validation')]
    return ledger,state,records,digest(experiment/'aggregate_training_integrity.json')


def analyze_seed(seed, branch_root, fusion_root, ledger, state, records, np):
    directory=fusion_root/f'final_seed{seed}'
    cache_path=directory/'validation_logits_evidence_local.npz'
    aggregate=read_json(directory/'aggregate_validation_fusion_scan.json')
    if (aggregate.get('status')!='completed' or aggregate.get('seed')!=seed
            or aggregate.get('test_images_or_predictions_read') is not False
            or aggregate.get('query_labels_used_in_matching') is not False):
        raise ValueError('Expected a completed validation-only fusion scan')
    if digest(cache_path)!=aggregate['cache_sha256']:
        raise ValueError('H cache changed after its final scan')
    with np.load(cache_path,allow_pickle=False) as data:
        z0,evidence,targets=data['z0'],data['evidence'],data['targets']
        query_ids=[str(x) for x in data['query_ids']]
        info=json.loads(str(data['metadata_json'].item()))
    classes=info['classes']; plan=info['plan']; inputs=ledger['inputs']
    if aggregate.get('classes')!=classes or info.get('selected_epoch')!=aggregate.get('checkpoint_selected_epoch'):
        raise ValueError('H cache class order/selected epoch differs from its aggregate scan')
    if len(classes)!=4 or len(set(classes))!=4 or targets.shape!=(227,) or targets.dtype.kind not in ('i','u'):
        raise ValueError('Invalid H classes/targets')
    if targets.min()<0 or targets.max()>=len(classes) or z0.shape!=(227,4) or evidence.shape!=z0.shape:
        raise ValueError('Invalid H target range/logit shapes')
    if z0.dtype!=np.float32 or evidence.dtype!=np.float32 or not np.isfinite(z0).all() or not np.isfinite(evidence).all():
        raise ValueError('Expected finite float32 H logits/evidence')
    if evidence.min()<-1e-6 or evidence.max()>1+1e-6:
        raise ValueError('H evidence is outside its bounded range')
    if plan['seed']!=seed or plan['manifest_sha256']!=inputs['manifest']['sha256'] or plan['code_sha256']!=inputs['code_sha256']:
        raise ValueError('H cache seed/manifest/code provenance differs')
    if plan['evidence_temperature']!=TEMPERATURE or tuple(plan['gamma_grid'])!=GAMMAS:
        raise ValueError('H scan temperature/grid differs from the fixed diagnostic grid')
    h_hash=state['steps'][f'seed{seed}/full_h']['output_sha256']
    if info['checkpoint_sha256']!=h_hash or aggregate['checkpoint_sha256']!=h_hash:
        raise ValueError('H cache does not come from the completed checkpoint')
    expected={str(r['id']):str(r['label']) for r in records if r['development_split'] in ('val','validation')}
    if len(expected)!=227 or len(set(query_ids))!=227 or {sid:classes[int(y)] for sid,y in zip(query_ids,targets)}!=expected:
        raise ValueError('H query IDs/labels differ from the pinned validation manifest')
    natural={}; file_hashes={}; source_hashes={}; normalized_configs={}
    all_expected={str(r['id']):('train' if r['development_split']=='train' else 'validation',str(r['label'])) for r in records}
    for arm in ('baseline','disentangle'):
        path=branch_root/f'seed{seed}'/'branch_diagnostics_features_local'/f'{arm}_natural_features_local.npz'
        loaded=read_natural(path,np); meta=loaded['metadata']
        if meta.get('schema_version')!=1 or meta.get('seed')!=seed or meta.get('arm')!=arm or meta.get('encoder_stage')!='stage1':
            raise ValueError('Natural cache encoder stage/arm/seed differs')
        if meta.get('test_images_or_predictions_read') is not False or not meta.get('validation_checkpoint_replay',{}).get('passed'):
            raise ValueError('Original head replay or validation-only scope has not passed')
        if any(meta.get(key) is not True for key in ('frozen_encoder','hard_gate','natural_views_only')):
            raise ValueError('Expected frozen, hard-gated natural-view Stage1 extraction')
        if meta['manifest_sha256']!=inputs['manifest']['sha256']:
            raise ValueError('Natural cache manifest differs')
        if set(meta['code_sha256'])!={'stage1.py','data.py','geometry.py','contracts.py','cli.py'}:
            raise ValueError('Natural cache model code provenance is incomplete')
        for name,expected_hash in meta['code_sha256'].items():
            if inputs['code_sha256'].get('sar_port/two_stage/'+name)!=expected_hash:
                raise ValueError('Natural cache model code differs from H')
        source_hashes[arm]=state['steps'][f'seed{seed}/{arm}_stage1']['output_sha256']
        if meta['checkpoint_sha256']!=source_hashes[arm]:
            raise ValueError('Natural cache Stage1 checkpoint differs from the completed training artifact')
        measured={str(sid):(str(split),str(label)) for sid,split,label in zip(loaded['ids'],loaded['split'],loaded['label'])}
        if measured!=all_expected:
            raise ValueError('Natural cache identities/labels differ from the pinned train/validation manifest')
        natural[arm]=align_heads(loaded,query_ids,targets,classes,np)
        actual=metrics(targets,natural[arm].argmax(1),classes,np)
        metric_replay(actual,meta['checkpoint_validation_metrics'],arm+' checkpoint')
        metric_replay(actual,meta['head_metrics']['validation'],arm+' natural cache')
        file_hashes[arm]=digest(path)
        normalized_configs[arm]=(meta['input_config'],meta['geometry_config'])
    if normalized_configs['baseline']!=normalized_configs['disentangle']:
        raise ValueError('Baseline and disentangle input/geometry interfaces differ')
    metric_replay(metrics(targets,z0.argmax(1),classes,np),aggregate['h0'],'H h0 cache')
    baseline_pred=natural['baseline'].argmax(1)
    original=scan(z0,evidence,targets,classes,baseline_pred,np)
    by_gamma={row['gamma']:row for row in aggregate['scan']}
    for row in original:
        metric_replay(row,by_gamma[row['gamma']],'Original H fusion')
    replacement=scan(natural['disentangle'],evidence,targets,classes,baseline_pred,np)
    return dict(seed=seed,classes=classes,validation_count=227,
        stage1_baseline=metrics(targets,baseline_pred,classes,np),
        stage1_disentangle=metrics(targets,natural['disentangle'].argmax(1),classes,np),
        branches=dict(original_h0=original,stage1_disentangle_head=replacement),
        provenance=dict(natural_npz_sha256=file_hashes,h_cache_sha256=digest(cache_path),
            final_fusion_scan_sha256=digest(directory/'aggregate_validation_fusion_scan.json'),
            stage1_checkpoint_sha256=source_hashes,full_h_checkpoint_sha256=h_hash,
            manifest_sha256=inputs['manifest']['sha256'],
            validation_id_set_sha256=hashlib.sha256(json.dumps(sorted(query_ids)).encode()).hexdigest(),
            encoder_stage1_to_base_to_h_exact=True,original_head_checkpoint_metrics_replayed=True))


def summarize(per_seed):
    def describe(values):
        return dict(mean=statistics.mean(values),seed_sd=statistics.stdev(values),n_seeds=len(values))
    if any(r['classes']!=per_seed[0]['classes'] or r['provenance']['validation_id_set_sha256']!=per_seed[0]['provenance']['validation_id_set_sha256']
           for r in per_seed):
        raise ValueError('Different class mappings/validation samples across seeds')
    result={}
    for branch in ('original_h0','stage1_disentangle_head'):
        result[branch]=[]
        for index,gamma in enumerate(GAMMAS):
            rows=[seed['branches'][branch][index] for seed in per_seed]
            result[branch].append(dict(gamma=gamma,
                metrics={key:describe([r[key] for r in rows]) for key in METRICS},
                rescued_vs_own_gamma0=describe([r['rescued_vs_own_gamma0'] for r in rows]),
                harmed_vs_own_gamma0=describe([r['harmed_vs_own_gamma0'] for r in rows]),
                delta_vs_stage1_baseline_pp={key:describe([r['delta_vs_stage1_baseline_pp'][key] for r in rows]) for key in METRICS}))
    return dict(branches=result,stage1_baseline={key:describe([r['stage1_baseline'][key] for r in per_seed]) for key in METRICS},
                stage1_disentangle={key:describe([r['stage1_disentangle'][key] for r in per_seed]) for key in METRICS})


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--branch-root',type=Path,default=ROOT)
    parser.add_argument('--fusion-root',type=Path,default=ROOT.parent/'oss2_h_tuning_20261122')
    parser.add_argument('--experiment-root',type=Path,default=ROOT.parent/'oss2_three_arm_experiment_20261122')
    parser.add_argument('--output',type=Path,default=ROOT/'aggregate_head_reuse_validation.json')
    args=parser.parse_args(argv)
    if args.output.exists(): raise ValueError('Existing head-reuse report preserved; refusing overwrite')
    ledger,state,records,integrity_hash=readiness(args.experiment_root,args.branch_root,args.fusion_root)
    import numpy as np
    per_seed=[analyze_seed(seed,args.branch_root,args.fusion_root,ledger,state,records,np) for seed in SEEDS]
    result=dict(status='completed',created_at=datetime.now(timezone.utc).isoformat(),split='validation',
        purpose='Readout replacement on fixed previously-trained H evidence; not a retrained H candidate',
        seeds=list(SEEDS),gamma_grid=list(GAMMAS),evidence_temperature=TEMPERATURE,
        test_images_or_predictions_read=False,training_or_model_forward_performed=False,
        query_labels_used_for_matching=False,integrity_report_sha256=integrity_hash,
        script_sha256=digest(Path(__file__)),per_seed=per_seed,aggregate=summarize(per_seed),
        interpretation=[
            'H parameters and evidence are held fixed; only the additive base logits are substituted.',
            'H was trained with its original h0. Replacing h0 changes the readout calibration and does not test retraining H with the Stage1 head.',
            'Gamma=0 reports the corresponding head alone; the Stage1 baseline remains a fixed comparator.',
            'The same validation ships are reused across seeds and for checkpoint selection; this is development diagnosis, not independent confirmation.',
            'No common gamma is selected or written by this diagnostic; the original selection files remain unchanged.'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_suffix('.json.tmp')
    with temporary.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    temporary.replace(args.output)
    print(json.dumps(dict(status='completed',output=str(args.output),test_access=False)))


if __name__=='__main__':
    try:
        main()
    except (OSError,ValueError,KeyError,TypeError) as exc:
        print('STOP: '+str(exc))
        raise SystemExit(1)
