"""Single-seed exploratory H fusion scan; train bank and validation only.

No training, GPU, test images, test predictions, or original-file writes.
The fixed grid is declared before inference and cached E is reused at all gamma.
"""
import argparse
import ctypes
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parent
EXPERIMENT = ROOT.parent / 'oss2_three_arm_experiment_20261122'
STATUS_ROOT = ROOT
GAMMAS = (0., .25, .5, 1., 2., 4.)
TEMPERATURE = .2
START_FREE_GIB = 2.0
RUN_FREE_GIB = .75


class MemoryPressure(RuntimeError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def memory():
    class Status(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in ('total', 'available', 'page_total',
                                                  'page_available', 'virtual_total',
                                                  'virtual_available', 'extended')]
    status = Status()
    status.length = ctypes.sizeof(Status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError('Cannot check Windows physical memory')
    return dict(available_gib=status.available / 2**30, total_gib=status.total / 2**30,
                percent_used=int(status.load))


def guard_memory(minimum, phase):
    status = memory()
    if status['available_gib'] < minimum:
        raise MemoryPressure(f'{phase}: {status["available_gib"]:.3f} GiB free; '
                             f'at least {minimum:.2f} GiB required; training left untouched')
    return status


def stamp(path):
    info = Path(path).stat()
    return dict(size=info.st_size, mtime_ns=info.st_mtime_ns, ctime_ns=info.st_ctime_ns)


def pin_inputs(seed=17, output_dir=ROOT):
    ledger = read_json(EXPERIMENT / 'ledger_local.json')
    manifest = EXPERIMENT / 'experiment_manifest_local.json'
    if digest(manifest) != ledger['inputs']['manifest']['sha256']:
        raise RuntimeError('Original manifest hash differs from its training ledger')
    code_root = Path(ledger['inputs']['code_root'])
    for relative, expected in ledger['inputs']['code_sha256'].items():
        if digest(code_root / relative) != expected:
            raise RuntimeError(f'Original training code changed: {relative}')
    plan = dict(seed=seed, purpose='single-seed validation exploration, not confirmatory evidence',
                gamma_grid=list(GAMMAS), evidence_temperature=TEMPERATURE, device='cpu',
                torch_threads=1, inference_query_batch_size=1, expected_validation_count=227,
                expected_reference_count=32, expected_train_count=718,
                manifest_sha256=ledger['inputs']['manifest']['sha256'],
                code_sha256=ledger['inputs']['code_sha256'],
                query_labels_used_in_matching=False, test_images_or_predictions_read=False)
    path = output_dir / 'scan_plan.json'
    if path.exists():
        if read_json(path) != plan:
            raise RuntimeError('Existing exploratory plan differs; refusing to change the grid or inputs')
    else:
        write_json(path, plan)
    return code_root, plan


def snapshot(seed=17, output_dir=ROOT, checkpoint=None):
    """Buffered disk copy, no Torch import; never refresh an already pinned snapshot."""
    source = (Path(checkpoint) if checkpoint is not None else
              EXPERIMENT / 'runs' / f'seed{seed}' / 'full_h.pt').resolve()
    target = output_dir / 'full_h_snapshot_local.pt'
    record_path = output_dir / 'snapshot_local.json'
    if target.exists() or record_path.exists():
        if not target.exists() or not record_path.exists():
            raise RuntimeError('Incomplete snapshot pair; preserve it for manual inspection')
        record = read_json(record_path)
        if Path(record['source']).resolve() != source:
            raise RuntimeError('Output directory already pins another checkpoint source; use a new output directory')
        if digest(target) != record['sha256']:
            raise RuntimeError('Immutable H snapshot hash changed')
        return target, record
    temporary = target.with_suffix('.pt.copying')
    for attempt in range(3):
        before = stamp(source)
        with source.open('rb') as incoming, temporary.open('wb') as outgoing:
            shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
        after = stamp(source)
        if before == after and temporary.stat().st_size == before['size']:
            os.replace(temporary, target)
            record = dict(created_at=now(), source=str(source), source_stat_before=before,
                          source_stat_after=after, stat_unchanged_during_copy=True,
                          sha256=digest(target), selected_epoch=None,
                          selected_epoch_status='read from checkpoint only after memory guard permits Torch')
            write_json(record_path, record)
            return target, record
        time.sleep(.25)
    raise RuntimeError('Best checkpoint changed during all three copies; no snapshot accepted')


def distribution(values, np):
    values = np.asarray(values, dtype=float)
    return dict(count=int(values.size), mean=float(values.mean()) if values.size else None,
                quantiles={str(q): float(np.quantile(values, q)) if values.size else None
                           for q in (0., .1, .25, .5, .75, .9, 1.)})


def metrics(y, pred, classes, np):
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    np.add.at(matrix, (y, pred), 1)
    support, predicted = matrix.sum(1), matrix.sum(0)
    correct = matrix.diagonal()
    recall = np.divide(correct, support, out=np.zeros(len(classes)), where=support > 0)
    precision = np.divide(correct, predicted, out=np.zeros(len(classes)), where=predicted > 0)
    f1 = np.divide(2 * correct, support + predicted, out=np.zeros(len(classes)), where=support + predicted > 0)
    return dict(count=int(len(y)), accuracy=float(correct.sum() / len(y)),
                balanced_accuracy=float(recall[support > 0].mean()), macro_f1=float(f1.mean()),
                confusion_matrix=matrix.tolist(), per_class={str(c): dict(count=int(support[i]),
                    recall=float(recall[i]), precision=float(precision[i]), f1=float(f1[i]))
                    for i, c in enumerate(classes)})


def aggregate_cache(cache_path, plan, record, np):
    with np.load(cache_path, allow_pickle=False) as data:
        z0, evidence, y = data['z0'], data['evidence'], data['targets']
        support = data['support']
        info = json.loads(str(data['metadata_json'].item()))
    if info['checkpoint_sha256'] != record['sha256'] or info['plan'] != plan:
        raise RuntimeError('Cache provenance differs from pinned snapshot/plan')
    classes = info['classes']
    if z0.shape != (227, len(classes)) or evidence.shape != z0.shape or support.shape != z0.shape:
        raise RuntimeError('Incomplete or malformed validation cache')
    if y.shape != (227,) or y.min() < 0 or y.max() >= len(classes):
        raise RuntimeError('Malformed cached validation targets')
    if not np.isfinite(z0).all() or not np.isfinite(evidence).all():
        raise RuntimeError('Nonfinite cached logits/evidence')
    if evidence.min() < -1e-6 or evidence.max() > 1 + 1e-6:
        raise RuntimeError('E outside [0,1]: cannot apply the gamma/Te margin bound')
    index = np.arange(len(y))
    pred0, pred_e = z0.argmax(1), evidence.argmax(1)
    correct0, correct_e = pred0 == y, pred_e == y
    nontrue = z0.copy()
    nontrue[index, y] = -np.inf
    rival = nontrue.argmax(1)
    signed_gap = nontrue[index, rival] - z0[index, y]
    wrong = ~correct0
    baseline = metrics(y, pred0, classes, np)
    results = []
    for gamma in GAMMAS:
        bound = gamma / TEMPERATURE
        logits = z0 + np.float32(bound) * evidence
        pred = logits.argmax(1)
        correct = pred == y
        rescue, harm = wrong & correct, correct0 & ~correct
        impossible = wrong & (signed_gap > bound)
        protected = correct0 & (-signed_gap > bound)
        inside = wrong & (signed_gap < bound)
        equality = wrong & (signed_gap == bound)
        # Strict bound is only a necessary feasibility condition; equality can at most tie.
        violations = int((rescue & impossible).sum() + (harm & protected).sum())
        if violations:
            raise RuntimeError('Observed correction violates the E in [0,1] bound')
        measured = metrics(y, pred, classes, np)
        measured.update(gamma=gamma, pairwise_logit_bound=bound,
            delta_vs_h0={name: measured[name] - baseline[name]
                         for name in ('accuracy', 'balanced_accuracy', 'macro_f1')},
            rescued=int(rescue.sum()), harmed=int(harm.sum()), net_correct=int(rescue.sum()-harm.sum()),
            h0_wrong_count=int(wrong.sum()), h0_wrong_margin_above_bound=int(impossible.sum()),
            h0_wrong_margin_strictly_inside_bound=int(inside.sum()),
            h0_wrong_margin_exactly_at_bound=int(equality.sum()),
            wrong_above_bound_fraction=float(impossible.sum()/max(1, wrong.sum())),
            h0_wrong_evidence_alone_correct=int((wrong & correct_e).sum()),
            h0_wrong_evidence_correct_but_above_bound=int((impossible & correct_e).sum()),
            h0_wrong_evidence_correct_inside_bound=int((inside & correct_e).sum()),
            h0_correct_protected_by_bound=int(protected.sum()), bound_invariant_violations=violations,
            rescue_harm_by_class={str(c): dict(rescued=int((rescue & (y==i)).sum()),
                harmed=int((harm & (y==i)).sum())) for i,c in enumerate(classes)},
            margin_distributions=dict(h0_wrong=distribution(signed_gap[wrong], np),
                rescued=distribution(signed_gap[rescue], np),
                harmed_h0_correct_margin=distribution(-signed_gap[harm], np),
                unrescued=distribution(signed_gap[wrong & ~correct], np)),
            actual_evidence_correction_vs_h0_rival=distribution(
                bound*(evidence[index,y]-evidence[index,rival]), np))
        results.append(measured)
    best_ba = max(r['balanced_accuracy'] for r in results)
    return dict(created_at=now(), status='completed', seed=plan['seed'],
        scope='single-seed validation fusion exploration; not a confirmation result',
        checkpoint_sha256=record['sha256'], checkpoint_selected_epoch=info['selected_epoch'],
        checkpoint_selected_validation_metrics=info['selected_validation_metrics'],
        manifest_sha256=plan['manifest_sha256'], cache_sha256=digest(cache_path),
        classes=classes, train_count=718, reference_count=32, validation_count=227,
        device='cpu', torch_threads=1, gamma_grid=list(GAMMAS), evidence_temperature=TEMPERATURE,
        query_labels_used_in_matching=False, test_images_or_predictions_read=False,
        h0=baseline, evidence_alone=metrics(y,pred_e,classes,np),
        true_class_has_reference_support=int(support[index,y].sum()),
        scan=results, gamma_tied_for_highest_validation_ba=[r['gamma'] for r in results
                                                        if r['balanced_accuracy']==best_ba],
        interpretation=[
            'All gamma values reuse exactly the same frozen h0 logits and H evidence.',
            'E-alone correctness establishes evidence ranking, not sufficient evidence margin.',
            'A wrong h0 margin greater than gamma/Te cannot be repaired at this fusion strength.',
            'A margin below the bound permits but does not guarantee repair.',
            'The checkpoint and gamma are explored on the same validation split: optimistic selection is possible.',
            'Do not treat single-seed tuning gains as replicated gains or use test data to tune.'])


def create_cache(code_root, plan, snapshot_path, record, cache_path, output_dir=ROOT):
    # Import order is deliberate on this Windows environment.
    guard_memory(START_FREE_GIB, 'before NumPy/Torch import')
    os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                      CUDA_VISIBLE_DEVICES='', SAR_PORT_PREPARE_CACHE=str(EXPERIMENT/'prepared_cache_local'))
    sys.path.insert(0, str(code_root))
    import numpy as np
    import torch
    from sar_port.two_stage.cli import load_checkpoint, prepare, restore_encoder, encode_samples, metadata
    from sar_port.two_stage.data import InputConfig
    from sar_port.two_stage.geometry import GeometryConfig
    from sar_port.two_stage.h import HModel, HConfig
    from sar_port.protocol import legal_reference_mask
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    guard_memory(RUN_FREE_GIB, 'before checkpoint load')
    saved = load_checkpoint(snapshot_path)
    if saved['stage'] != 'h' or abs(saved['h_config']['evidence_temperature']-TEMPERATURE)>1e-12:
        raise RuntimeError('Expected H checkpoint with Te=0.2')
    if saved.get('h_training_config', {}).get('seed') != plan['seed']:
        raise RuntimeError('Checkpoint training seed differs from --seed')
    selected = [r for r in saved['h_history'] if r['stage']=='match'][-1]
    record.update(selected_epoch=selected['epoch'], selected_epoch_status='verified from snapshot h_history',
                  selected_validation_metrics=selected['validation_metrics'])
    write_json(output_dir/'snapshot_local.json', record)
    raw = read_json(EXPERIMENT/'experiment_manifest_local.json')
    raw = raw if isinstance(raw, list) else raw['records']
    # Manifest metadata are read only to resolve fixed membership; other split records are discarded.
    rows = [dict(r, label=str(r['label'])) for r in raw if r.get('development_split') in ('train','val','validation')]
    del raw
    by_id = {r['id']:i for i,r in enumerate(rows)}
    train = [i for i,r in enumerate(rows) if r['development_split']=='train']
    validation = [i for i,r in enumerate(rows) if r['development_split'] in ('val','validation')]
    if len(by_id)!=len(rows) or len(train)!=718 or len(validation)!=227:
        raise RuntimeError('Pinned train/validation split size or identity uniqueness changed')
    ids = {r['id'] for r in rows}
    if metadata(rows) != [r for r in saved['manifest_metadata'] if r['id'] in ids]:
        raise RuntimeError('Checkpoint train/validation metadata differ from pinned manifest')
    if saved['source_ids'] != [rows[i]['id'] for i in train]:
        raise RuntimeError('Checkpoint source split differs')
    if len(saved['reference_ids'])!=32 or len(set(saved['reference_ids']))!=32:
        raise RuntimeError('Expected actual 32 distinct references from this checkpoint')
    if not set(saved['reference_ids'])<=set(saved['source_ids']):
        raise RuntimeError('Reference bank contains a nontraining record')
    ref_indices = [by_id[sid] for sid in saved['reference_ids']]
    ref_rows = [rows[i] for i in ref_indices]
    for key in ('group_id','vessel_id'):
        if {rows[i][key] for i in train} & {rows[i][key] for i in validation}:
            raise RuntimeError(f'Train/validation overlap in {key}')
    input_config, geometry_config = InputConfig(**saved['input_config']), GeometryConfig(**saved['geometry_config'])
    classes = saved['labels']; label_map = {v:i for i,v in enumerate(classes)}
    guard_memory(RUN_FREE_GIB, 'before restoring frozen encoder')
    encoder = restore_encoder(saved, 'cpu').freeze_for_h()
    # Release saved tensor copies before preparing samples; no second base checkpoint is loaded.
    del saved['encoder']
    h = HModel(HConfig(**saved['h_config']))
    h.load_state_dict(saved['h']); h.set_stage('match').eval(); h.requires_grad_(False)
    del saved
    gc.collect()
    reference_features = {}
    with torch.inference_mode():
        for i in ref_indices:
            guard_memory(RUN_FREE_GIB, 'reference encoding')
            sample = prepare(rows, EXPERIMENT, input_config, geometry_config, [i])
            reference_features.update(encode_samples(encoder, sample, [i], 'cpu', 1))
            del sample
        refs = [reference_features[i] for i in ref_indices]
        ref_labels = torch.tensor([label_map[r['label']] for r in ref_rows])
        z0, evidence, support, targets, query_ids = [], [], [], [], []
        for position, i in enumerate(validation):
            guard_memory(RUN_FREE_GIB, 'validation inference')
            sample = prepare(rows, EXPERIMENT, input_config, geometry_config, [i])
            query = encode_samples(encoder, sample, [i], 'cpu', 1)[i]
            identity = {key:rows[i][key] for key in ('vessel_id','group_id')}
            legal = torch.as_tensor(legal_reference_mask([identity], ref_rows))
            output = h([query], refs, ref_labels, [r['port'] for r in ref_rows], legal)
            torch.testing.assert_close(output['logits'], output['z0'] + h.config.gamma/TEMPERATURE*output['evidence'])
            z0.append(output['z0'][0].cpu().numpy().copy())
            evidence.append(output['evidence'][0].cpu().numpy().copy())
            support.append(output['support'][0].cpu().numpy().copy())
            # Query truth is consulted only after the complete label-blind H forward.
            targets.append(label_map[rows[i]['label']]); query_ids.append(rows[i]['id'])
            del sample, query, output
            if (position+1)%10==0 or position+1==len(validation):
                write_json(output_dir/'scan_status.json', dict(status='running', phase='validation_inference',
                    completed=position+1, total=len(validation), updated_at=now(), memory=memory()))
                print(json.dumps(dict(validation_completed=position+1,total=len(validation))), flush=True)
    info = dict(plan=plan, checkpoint_sha256=record['sha256'], classes=classes,
                selected_epoch=record['selected_epoch'], selected_validation_metrics=selected['validation_metrics'])
    temporary = cache_path.with_suffix('.npz.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, z0=np.stack(z0), evidence=np.stack(evidence), support=np.stack(support),
            targets=np.asarray(targets,dtype=np.int64), query_ids=np.asarray(query_ids),
            metadata_json=np.asarray(json.dumps(info, ensure_ascii=False)))
    os.replace(temporary, cache_path)
    del h, encoder, refs, reference_features
    gc.collect()
    return np


def main(argv=None):
    global STATUS_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=(17, 29, 43), default=17)
    parser.add_argument('--output-dir', type=Path, default=ROOT,
                        help='Independent immutable snapshot/cache directory; default is this script directory')
    parser.add_argument('--checkpoint', type=Path,
                        help='Optional H checkpoint; default is original runs/seed{seed}/full_h.pt')
    parser.add_argument('--snapshot-only', action='store_true', help='Pin plan and stable checkpoint without NumPy/Torch')
    parser.add_argument('--analyze-cache', action='store_true', help='Require existing private cache; never load Torch')
    args = parser.parse_args(argv)
    if args.snapshot_only and args.analyze_cache:
        parser.error('Choose only one mode')
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    STATUS_ROOT = output_dir
    code_root, plan = pin_inputs(args.seed, output_dir)
    snapshot_path, record = snapshot(args.seed, output_dir, args.checkpoint)
    if args.snapshot_only:
        result = dict(status='snapshot_ready', seed=args.seed, created_at=now(), snapshot_sha256=record['sha256'],
                      selected_epoch=record.get('selected_epoch'), memory=memory(),
                      torch_imported=False, inference_started=False)
        write_json(output_dir/'scan_status.json', result)
        print(json.dumps(result, ensure_ascii=False)); return
    cache_path = output_dir/'validation_logits_evidence_local.npz'
    if cache_path.exists():
        guard_memory(.25, 'cache-only NumPy analysis')
        os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        import numpy as np
    else:
        if args.analyze_cache:
            raise RuntimeError('No complete private validation cache exists')
        np = create_cache(code_root, plan, snapshot_path, record, cache_path, output_dir)
    result = aggregate_cache(cache_path, plan, record, np)
    write_json(output_dir/'aggregate_validation_fusion_scan.json', result)
    write_json(output_dir/'scan_status.json', dict(status='completed', seed=args.seed, completed_at=now(),
        checkpoint_selected_epoch=result['checkpoint_selected_epoch'], memory=memory()))
    print(json.dumps(dict(status='completed', seed=args.seed, selected_epoch=result['checkpoint_selected_epoch'],
        h0_ba=result['h0']['balanced_accuracy'],
        scan=[{k:r[k] for k in ('gamma','balanced_accuracy','accuracy','macro_f1','rescued','harmed')}
              for r in result['scan']]), ensure_ascii=False), flush=True)


if __name__=='__main__':
    try:
        main()
    except MemoryPressure as exc:
        result = dict(status='paused_memory_pressure', updated_at=now(), reason=str(exc),
            memory=memory(), torch_imported='torch' in sys.modules,
            instruction='Retry the same command after training coordination frees memory; fixed snapshot is reused.')
        write_json(STATUS_ROOT/'scan_status.json', result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        sys.exit(2)
    except (OSError,ValueError,RuntimeError,KeyError,IndexError) as exc:
        write_json(STATUS_ROOT/'scan_status.json', dict(status='failed', updated_at=now(), reason=str(exc)))
        print(f'STOP: {exc}', file=sys.stderr)
        sys.exit(1)
