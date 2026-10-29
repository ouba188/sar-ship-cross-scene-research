"""Minimal final-model baseline/ablation fits, own selection, adaptation and evaluation."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT, WORK = HERE.parent, HERE.parents[2]
RELEASE = WORK/'sar_m123_current_20261027'
FIT = ROOT/'spatial_m12_refit_20261029'
FULL = ROOT/'spatial_m12_full_20261029'
sys.path.insert(0, str(WORK/'sar_m123_standalone'))
from sar_m123.current import training
from sar_m123.current.model import setup, training_policy, reference_bank, forward_batch, LIVE, load_model
from sar_m123.current.data import read_manifest, prepare_record, batch, pair_mask, FREQUENCY_KEYS
from sar_m123.current.interventions import variants, select_teacher, select_records
from sar_m123.pool_replacement import PoolReplacementResNet50
from model import ARMS, AblationModel, fresh, load, save, fitting_loss

spec = importlib.util.spec_from_file_location('paired_source_refit', FIT/'run.py')
refit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(refit)
SEEDS = (17, 29, 43)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def rows_all():
    return json.loads((FULL/'cohorts.json').read_text(encoding='utf-8'))


def inputs():
    rows = {k: v for k, v in rows_all().items() if k in ('fit', 'development', 'references')}
    table, arrays = refit.cached.load()
    lookup = {str(int(v)): i for i, v in enumerate(table.obs_id)}
    indices = {k: np.array([lookup[str(r['id'])] for r in group]) for k, group in rows.items()}
    receipt = json.loads((FIT/'cache/complete.json').read_text(encoding='utf-8'))
    assert sorted(set(np.concatenate(list(indices.values())).tolist())) == receipt['original_indices']
    for group, ix in indices.items():
        for r, i in zip(rows[group], ix):
            assert r['label'] == int(table.iloc[i].class_id)
            assert str(r['mmsi']) == str(table.iloc[i].mmsi)
    arrays['prefix'] = np.load(FIT/'cache/prefix.npy', mmap_mode='r')
    arrays['masks'] = np.load(FIT/'cache/masks.npy', mmap_mode='r')
    refit.gpu_cache(arrays, indices)
    values, extra = refit.get_batch(arrays, indices['references'])
    bank = dict(zip(('prefix', 'masks', 'valid', 'available'), values), **extra)
    return rows, arrays, indices, bank


def check(rows, arrays, indices, bank):
    chosen = [0, 1, 2]
    q = [rows['development'][i] for i in chosen]
    ids = indices['development'][chosen]
    checks = {}
    # Compare nonzero trained operators against direct removals of the real full model.
    path = FIT/'training/spatial_s17/model.pt'
    for arm in ('full', *ARMS):
        original = load_model(path, 'cuda')
        candidate = AblationModel(arm).cuda().eval()
        source = original.state_dict()
        state = {k: source[k] for k in candidate.state_dict()}
        if arm in ('plain', 'no_m2'):
            w = source['head.weight']
            state['head.weight'] = torch.cat((w[:, :2048], w[:, 3584:5632], w[:, -2:]), 1)
        candidate.load_state_dict(state)
        if arm == 'no_m1':
            for layer in original.internal_m1:
                layer.gamma.data.zero_()
        elif arm == 'no_m2':
            for layer in original.internal_m2:
                layer.enabled = False
            original.head.weight.data[:, 2048:3584].zero_()
            original.head.weight.data[:, 5632:7168].zero_()
        elif arm == 'no_m3':
            original.m3 = nn.ModuleList()
        if arm == 'plain':
            ordinary = PoolReplacementResNet50(None, mode='native').cuda().eval()
            ordinary.backbone.load_state_dict(candidate.backbone.state_dict())
            ordinary.head.load_state_dict(candidate.head.state_dict())
            with torch.no_grad():
                values, _ = refit.get_batch(arrays, ids)
                expected = ordinary.forward_from(*values)
            del ordinary
        else:
            with torch.no_grad():
                expected = refit.score(original, arrays, ids, q, rows['references'], bank)
        with torch.no_grad():
            actual = refit.score(candidate, arrays, ids, q, rows['references'], bank)
        error = float(abs(actual-expected).max())
        assert error < 1e-4, (arm, error)
        del original, candidate, actual, expected
        if arm == 'full':
            checks[arm] = dict(nonzero_path_replay_error=error)
            continue
        net = fresh(arm, RELEASE/'weights/init.pt')
        training_policy(net, 'fit')
        score, detail = refit.score(net, arrays, ids, q, rows['references'], bank, True)
        loss, terms = fitting_loss(score, detail, torch.tensor([r['label'] for r in q], device='cuda'),
            torch.ones(len(q), device='cuda'), torch.tensor([r['label'] for r in rows['references']], device='cuda'))
        loss.backward()
        grads = {n: float(p.grad.norm()) for n, p in net.named_parameters() if p.grad is not None}
        assert all(grads.get(n, 0) > 0 for n in LIVE)
        for module, prefix in (('m1','internal_m1.'), ('m2','internal_m2.'), ('m3','m3.')):
            active = arm not in ('plain', 'no_'+module)
            assert any(n.startswith(prefix) and v > 0 for n, v in grads.items()) == active, (arm, module)
        assert len(detail['aux_logits']) == (0 if arm in ('plain', 'no_m1') else 1)
        assert len(detail['relations']) == (0 if arm in ('plain', 'no_m3') else 2)
        saved = HERE/'checks'/f'{arm}.pt'
        save(net, saved, smoke_only=True)
        restored = load(saved)
        with torch.no_grad():
            replay = refit.score(restored, arrays, ids, q, rows['references'], bank)
        saved_error = float(abs(replay-score.detach()).max())
        assert saved_error < 1e-5
        # Native input agreement also covers the new head/readout path.
        native = batch([prepare_record(r) for r in q], 'cuda')
        refs = reference_bank(net, rows['references'], 'cuda')
        with torch.no_grad():
            direct = forward_batch(net, native, refs, pair_mask(q, rows['references'], 'cuda'))
        native_error = float(abs(direct-score.detach()).max())
        assert native_error < 1e-4, (arm, native_error)
        checks[arm] = dict(nonzero_path_replay_error=error, save_replay_error=saved_error,
            native_cache_error=native_error, gradients=grads,
            trainable=sum(p.numel() for p in net.parameters() if p.requires_grad),
            auxiliary_count=len(detail['aux_logits']), relation_count=len(detail['relations']))
        del net, restored, score, loss, detail, native, refs, replay, direct
        torch.cuda.empty_cache()
    write(HERE/'checks.json', checks)
    print('CHECKS_PASS', flush=True)


def fit(rows, arrays, indices, bank, arms, seeds):
    # Reuse the exact paired source-fit loop; only model/removal-aware loss and serialization change.
    refit.HERE = HERE/'source_fit'
    refit.HERE.mkdir(parents=True, exist_ok=True)
    refit.model = lambda arm: fresh(arm, RELEASE/'weights/init.pt')
    refit.fitting_loss = fitting_loss
    refit.save_model, refit.load_model = save, load
    refit.train(rows, arrays, indices, bank, arms=arms, seeds=seeds)


def checkpoint(arm, seed, stage):
    return (HERE/'source_fit' if stage == 'before' else HERE)/'training'/f'{arm}_s{seed}'/'model.pt'


@torch.no_grad()
def select(rows, arms, seeds):
    models = {(a, s): load(checkpoint(a, s, 'before')) for s in seeds for a in arms}
    first = next(iter(models.values()))
    bank = reference_bank(first, rows['references'], 'cuda')
    results = {key: [] for key in models}
    tick = time.perf_counter()
    for i, record in enumerate(rows['adapt']):
        items, meta = variants(record)
        scores = {key: [] for key in models}
        for start in range(0, len(items), 8):
            values = batch(items[start:start+8], 'cuda')
            prefix = first.prefix(values['images'], stop=2)
            legal = pair_mask([record]*len(values['images']), rows['references'], 'cuda')
            for key, net in models.items():
                if key[0] == 'plain' and start > 0:
                    continue
                scores[key].append(net.forward_from(prefix, **{k:v for k,v in values.items() if k!='images'},
                    reference=bank, pair_valid=legal).cpu().numpy())
        y = record['label']
        for key, net in models.items():
            score = np.concatenate(scores[key])
            original = torch.tensor(score[0])
            if key[0] == 'plain':
                results[key].append(dict(id=record['id'], label=y, original=original.tolist(), useful=False))
                continue
            choice = select_teacher(score, meta, y)
            altered = {k:v.copy() for k,v in items[choice['chosen_index']].items()}
            for k in FREQUENCY_KEYS:
                altered[k] = items[0][k]
            changed = forward_batch(net, batch([altered], 'cuda'), bank,
                pair_mask([record], rows['references'], 'cuda'))[0].cpu()
            def margin(z):
                other = z.clone()
                other[y] = -torch.inf
                return z[y]-other.max()
            useful = choice['useful'] and bool((changed.log_softmax(0)[y] > original.log_softmax(0)[y])
                                              & (margin(changed) > margin(original)))
            results[key].append(dict(id=record['id'], label=y, original=original.tolist(),
                useful=useful, altered=changed.tolist(), choice=choice,
                selected_intervention=meta[choice['chosen_index']]))
        if i % 16 == 0 or i+1 == len(rows['adapt']):
            print('SELECT', i+1, len(rows['adapt']), round(time.perf_counter()-tick,1), flush=True)
    checks = {}
    for a in arms:
        if a == 'plain':
            continue
        key = (a, seeds[0])
        expected = select_records(models[key], rows['adapt'][:1], rows['references'], 'cuda')[0]
        actual = results[key][0]
        assert expected['choice'] == actual['choice'] and expected['useful'] == actual['useful']
        error = max(float(np.max(abs(np.array(actual[k])-expected[k]))) for k in ('original','altered'))
        assert error < 1e-4
        checks[a] = error
    for (a,s), selection in results.items():
        out = HERE/'selection'/f'{a}_s{s}.json'
        assert not out.exists(), out
        write(out, selection)
    write(HERE/'selection_check.json', dict(native_errors=checks, seconds=time.perf_counter()-tick,
        counts={f'{a}_s{s}':sum(r['useful'] for r in result) for (a,s),result in results.items()}))


def adapt(rows, arms, seeds):
    training.save_model = save
    for s in seeds:
        for a in arms:
            out = checkpoint(a,s,'after').parent
            assert not (out/'model.pt').exists(), out
            net = load(checkpoint(a,s,'before'))
            targets = json.loads((HERE/'selection'/f'{a}_s{s}.json').read_text(encoding='utf-8'))
            fixed = {k:v.cpu().clone() for k,v in net.state_dict().items() if k not in LIVE}
            initial = {k:v.cpu().clone() for k,v in net.state_dict().items() if k in LIVE}
            tick = time.perf_counter()
            receipt = training.train(net, rows['adapt'], rows['references'], out, 'cuda',
                selection=targets, seed=s, coefficient=0. if a=='plain' else .3683066666126251)
            assert receipt['steps'] == 54
            assert all(torch.equal(net.state_dict()[k].cpu(), v) for k,v in fixed.items())
            updates = {k:float((net.state_dict()[k].cpu()-v).norm()) for k,v in initial.items()}
            assert all(v>0 for v in updates.values())
            restored = load(out/'model.pt')
            bank = reference_bank(net, rows['references'], 'cuda')
            q = rows['development'][:16]
            values = batch([prepare_record(r) for r in q], 'cuda')
            legal = pair_mask(q,rows['references'],'cuda')
            with torch.no_grad():
                error = float(abs(forward_batch(net,values,bank,legal)-forward_batch(restored,values,bank,legal)).max())
            assert error<1e-5
            receipt.update(arm=a, frozen_unchanged=True, live_updates=updates,
                           save_replay_error=error, seconds=time.perf_counter()-tick)
            write(out/'training.json',receipt)
            print('ADAPT_COMPLETE',a,s,round(receipt['seconds'],1),flush=True)
            del net,restored,bank,values,fixed,initial
            torch.cuda.empty_cache()


@torch.no_grad()
def evaluate(rows, arms, seeds):
    models = {(a,stage,s):load(checkpoint(a,s,stage)) for s in seeds for a in arms for stage in ('before','after')}
    first = next(iter(models.values()))
    bank = reference_bank(first, rows['references'], 'cuda')
    replay, tick = {},time.perf_counter()
    for scope in ('development','expanded'):
        scores = {key:[] for key in models}
        for start in range(0,len(rows[scope]),16):
            q = rows[scope][start:start+16]
            values = batch([prepare_record(r) for r in q],'cuda')
            prefix = first.prefix(values['images'],stop=2)
            legal = pair_mask(q,rows['references'],'cuda')
            for key,net in models.items():
                result=net.forward_from(prefix,**{k:v for k,v in values.items() if k!='images'},reference=bank,pair_valid=legal)
                if scope=='development' and start==0:
                    error=float(abs(result-forward_batch(net,values,bank,legal)).max())
                    assert error<1e-5
                    replay['_'.join(map(str,key))]=error
                scores[key].append(result.cpu().numpy())
            if start%256==0 or start+16>=len(rows[scope]):
                print('EVAL',scope,min(start+16,len(rows[scope])),len(rows[scope]),round(time.perf_counter()-tick,1),flush=True)
        for (a,stage,s),parts in scores.items():
            out=HERE/'predictions'/f'{scope}_{a}_{stage}_s{s}.npz'
            out.parent.mkdir(exist_ok=True)
            score=np.concatenate(parts)
            assert np.isfinite(score).all()
            np.savez_compressed(out,score=score,ids=np.array([str(r['id']) for r in rows[scope]]),y=np.array([r['label'] for r in rows[scope]]))
    write(HERE/'evaluation_check.json',dict(native_replay=replay,seconds=time.perf_counter()-tick,
        models=len(models),default_changed=False))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=('check','fit','select','adapt','evaluate','all'))
    p.add_argument('--arms',nargs='+',choices=ARMS,default=list(ARMS))
    p.add_argument('--seeds',nargs='+',type=int,default=list(SEEDS))
    args=p.parse_args()
    setup('cuda')
    if args.command in ('check','fit','all'):
        rows,arrays,indices,bank=inputs()
        if args.command == 'check' or (args.command == 'all' and not (HERE/'checks.json').exists()):
            check(rows,arrays,indices,bank)
        if args.command in ('fit','all'):
            assert (HERE/'checks.json').exists()
            fit(rows,arrays,indices,bank,args.arms,args.seeds)
        del arrays,indices,bank
        torch.cuda.empty_cache()
    rows=rows_all()
    for name in ('select','adapt','evaluate'):
        if args.command in (name,'all'):
            globals()[name](rows,args.arms,args.seeds)
            torch.cuda.empty_cache()
    print('COMPLETE',args.command,flush=True)


if __name__=='__main__':
    main()
