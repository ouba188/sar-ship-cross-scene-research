"""Matched source-only controls. Default verifies code; training is explicit."""
from pathlib import Path
import argparse
import csv
import importlib.util
import json
import sys
import time

from objective import MATCHED_MODES, select_targets, matched_plan, fulfillment, margin

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('matched_intervention_recipe', ROOT/'run.py')
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)
torch, np = base.torch, base.np
OUT = ROOT/'matched_20261119'
BATCH = 8


def output_dir(variant='legacy'):
    return ROOT/'unified_20261119' if variant == 'unified' else OUT


def provenance(variant='legacy'):
    paths = [base.CHECKPOINT, ROOT/'objective.py', ROOT/'run.py', Path(__file__),
             base.PAPER/'pcea_routed_correction_20261118/split_local.json']
    if variant == 'unified':
        from unified_model import SOURCE_FILES
        paths.extend(SOURCE_FILES)
    return {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in paths}


def schedule(fit, references, steps, seed):
    legal = np.setdiff1d(fit, references)
    order = np.random.default_rng(seed+20261118).permutation(legal)[:steps*BATCH]
    if len(order) != steps*BATCH:
        raise ValueError('Budget exceeds distinct source-fit samples')
    codes = np.random.default_rng(seed+20261119).integers(8, size=steps)
    return order, codes


def build_bank(inputs, ids, codes, seed, variant='legacy'):
    teacher = base.model(variant).requires_grad_(False).eval()
    packs, clean, off, changed, controls, active, eligible = [], [], [], [], [], [], []
    tick = time.perf_counter()
    for step, code in enumerate(codes):
        batch = ids[step*BATCH:(step+1)*BATCH]
        target, detail = base.measure(teacher, inputs, batch, int(code), include_off=True)
        clean.append(target['clean'].cpu())
        off.append(torch.from_numpy(detail['clean_off']))
        changed.append(torch.from_numpy(detail['intervention_scores']))
        controls.append(torch.from_numpy(detail['control_scores']))
        # Recompute eligibility directly from recorded physical-region diagnostics.
        available = inputs.data(batch, 'small', int(code))['available'].cpu().numpy()
        eligible.append(torch.tensor([[bool(available[j,p] and row['pixels'] >= 4 and row['l2'] > 1e-6)
            for j,row in enumerate(rows)] for (p,_,_),rows in zip(base.CONDITIONS,detail['diagnostics'])]))
        active.append(torch.from_numpy(detail['active']))
        packs.append(detail['diagnostics'])
        print('measure', step+1, '/', len(codes), flush=True)
    labels = torch.as_tensor(inputs.y[ids], dtype=torch.long)
    clean, off, changed, controls = torch.cat(clean), torch.cat(off), torch.cat(changed,1), torch.cat(controls,1)
    active, eligible = torch.cat(active), torch.cat(eligible,1)
    targets = select_targets(clean, changed, controls, labels, eligible)
    targets.update(matched_plan(targets, labels, active, seed))
    del teacher
    torch.cuda.empty_cache()
    return dict(targets=targets, labels=labels, active=active, ids=torch.as_tensor(ids),
        codes=torch.as_tensor(codes), references=torch.as_tensor(inputs.groups['references']),
        clean_off=off, intervention_scores=changed, control_scores=controls,
        region_eligible=eligible, diagnostics=packs, provenance=provenance(variant), seed=seed,
        model_variant=variant,
        seconds=time.perf_counter()-tick, source_only=True,
        protocol='26 teacher forwards per batch: original on/off + 12 interventions + 12 controls')


def batch_targets(bank, start, stop, device='cuda'):
    # All losses consume sample-major entries; full response grids remain in bank.
    keys = ('use','selected','clean','chosen','relation_gain','competitor_weights','margin_gain',
            'matched','eligible','donor','measured','shuffled','fixed')
    return {key: bank['targets'][key][start:stop].to(device) for key in keys}


@torch.no_grad()
def paired_scores(net, inputs, ids, code=0):
    net.eval()
    data = inputs.data(ids, 'small', code)
    reference = net.reference(inputs.data(inputs.groups['references'], 'small'))
    net.exchange.disabled = True
    off, _ = base.forward_data(net, data, inputs, ids, reference)
    net.exchange.disabled = False
    on, _ = base.forward_data(net, data, inputs, ids, reference)
    return on.detach(), off.detach(), net.exchange.trace['active'].any(1).detach()


def source_rows(bank, start, on, off, active, targets, step, mode):
    labels = bank['labels'][start:start+len(on)].to(on.device)
    fields = fulfillment(on, off, labels, targets, active, mode)
    off0 = margin(bank['clean_off'][start:start+len(on)].to(on.device), labels)
    fields['off_minus_start'] = fields['off_margin']-off0
    fields.update(source_id=bank['ids'][start:start+len(on)], label=labels,
        selected_condition=targets['selected'], accepted=targets['use'],
        matched=targets['matched'], donor=targets['donor'], active=active)
    columns = {key: val.cpu().tolist() for key,val in fields.items()}
    return [dict(step=step, **{key: val[i] for key,val in columns.items()}) for i in range(len(on))]


def write_rows(path, rows, append=False):
    with path.open('a' if append else 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        if not append:
            writer.writeheader()
        writer.writerows(rows)


def prepare(seed, steps, variant='legacy'):
    out = output_dir(variant)
    dest = out/f'bank_s{seed}_n{steps}.pt'
    if dest.exists():
        raise FileExistsError('Reuse existing bank explicitly; do not overwrite: '+str(dest))
    inputs, fit, _ = base.runtime(variant)
    ids, codes = schedule(fit, inputs.groups['references'], steps, seed)
    bank = build_bank(inputs, ids, codes, seed, variant)
    out.mkdir(exist_ok=True)
    torch.save(bank, dest)
    print('BANK', dest, 'eligible', int(bank['targets']['eligible'].sum()),
          'matched', int(bank['targets']['matched'].sum()), flush=True)


def evaluate_pair(net, inputs, ids):
    net.exchange.disabled = False
    on = base.m.full.predict(net, inputs, ids, 'small')
    net.exchange.disabled = True
    off = base.m.full.predict(net, inputs, ids, 'small')
    net.exchange.disabled = False
    return on, off


def train(path, mode, variant='legacy'):
    bank = torch.load(path, map_location='cpu', weights_only=True)
    if bank.get('model_variant','legacy') != variant:
        raise ValueError('Teacher bank belongs to a different model variant')
    if bank['provenance'] != provenance(variant):
        raise ValueError('Teacher, split or measurement code changed; prepare a fresh bank')
    inputs, fit, inner = base.runtime(variant)
    ids, codes = bank['ids'].numpy(), bank['codes'].numpy()
    if not np.isin(ids, fit).all() or np.isin(ids, inputs.groups['references']).any():
        raise ValueError('Bank must contain only non-reference source-fit queries')
    if not np.array_equal(bank['references'].numpy(), inputs.groups['references']) or not np.array_equal(bank['labels'].numpy(), inputs.y[ids]):
        raise ValueError('Source identities/labels or reference bank changed')
    dest = output_dir(variant)/f'{path.stem}_{mode}'
    dest.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(bank['seed'])
    torch.cuda.manual_seed_all(bank['seed'])
    net = base.model(variant)
    groups = base.optimizer_groups(net, variant)
    opt = torch.optim.AdamW(groups, weight_decay=.01)
    initial_on, initial_off = evaluate_pair(net, inputs, inner)
    best = base.m.full.metrics(inputs.y[inner], initial_on)
    torch.save(net.state_dict(), dest/'best.pt')
    np.savez_compressed(dest/'inner_start_local.npz', ids=inner, label=inputs.y[inner], on=initial_on, off=initial_off)
    np.savez_compressed(dest/'inner_best_local.npz', ids=inner, label=inputs.y[inner], on=initial_on, off=initial_off)
    config = dict(mode=mode, bank=str(path), seed=bank['seed'], steps=len(codes), batch=BATCH,
        start=str(base.CHECKPOINT), rates={g['name']:g['lr'] for g in groups}, extra_weight=.2, fixed_gain=.1,
        model_variant=variant, selection_mode=getattr(net,'selection_mode',None),
        gain_routing=['ccrel','exchange'], classification='all existing trainable parameters',
        references=16, selection='source-inner BA/OA/-NLL including step0; previously exposed development set',
        external_evaluation=False, singleton_gain_policy='excluded equally in fixed/measured/shuffled')
    (dest/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    history = []
    for step,code in enumerate(codes):
        start = step*BATCH
        batch = ids[start:start+BATCH]
        targets = batch_targets(bank, start, start+BATCH)
        opt.zero_grad(set_to_none=True)
        _, _, stats = base.backward(net, inputs, batch, int(code), targets, mode)
        torch.nn.utils.clip_grad_norm_([p for group in groups for p in group['params']], 5.)
        opt.step()
        on, off, active = paired_scores(net, inputs, batch, int(code))
        rows = source_rows(bank, start, on, off, active, targets, step+1, mode)
        write_rows(dest/'source_progress_local.csv', rows, append=step>0)
        history.append(dict(step=step+1, **stats))
        if (step+1)%30 == 0 or step+1 == len(codes):
            on, off = evaluate_pair(net, inputs, inner)
            metrics = base.m.full.metrics(inputs.y[inner], on)
            off_metrics = base.m.full.metrics(inputs.y[inner], off)
            if base.recipe.previous.key(metrics) > base.recipe.previous.key(best):
                best = metrics
                torch.save(net.state_dict(), dest/'best.pt')
                np.savez_compressed(dest/'inner_best_local.npz', ids=inner, label=inputs.y[inner], on=on, off=off)
            record = dict(step=step+1, on=metrics, off=off_metrics, batches=len(history),
                selected=sum(row['selected'] for row in history),
                unpaired=sum(row['unpaired'] for row in history),
                improve_mean=float(np.mean([row['improve'] for row in history])),
                protect_mean=float(np.mean([row['protect'] for row in history])))
            (dest/f'step_{step+1}.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
            print(mode, step+1, metrics['oa'], metrics['ba'], flush=True)
    torch.save(net.state_dict(), dest/'last.pt')
    (dest/'training_log.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    (dest/'receipt.json').write_text(json.dumps(dict(completed=True, updates=len(codes), source_only=True)), encoding='utf-8')


def verify(variant='legacy'):
    out = output_dir(variant)
    inputs, fit, _ = base.runtime(variant)
    refs = set(inputs.groups['references'].tolist())
    ids = np.asarray([int(i) for label in range(8) for i in
        [j for j in fit if inputs.y[j] == label and j not in refs and bool(np.asarray(inputs.arrays['available'][j]).all())][:2]])
    if len(ids) != 16:
        raise ValueError('Verification needs two available source observations per class')
    # Nontrivial geometric augmentation is covered in the second batch.
    codes = np.array([0, 3])
    bank = build_bank(inputs, ids, codes, 17, variant)
    out.mkdir(exist_ok=True)
    torch.save(bank, out/'verification_bank_local.pt')
    net = base.model(variant)
    groups = base.optimizer_groups(net, variant)
    grouped = [id(p) for g in groups for p in g['params']]
    assert len(grouped) == len(set(grouped))
    assert set(grouped) == {id(p) for p in net.parameters() if p.requires_grad}
    reports, baseline_grads = {}, {}
    max_replay = 0.
    for mode in MATCHED_MODES:
        norms, errors, rows = [], [], []
        for step, code in enumerate(codes):
            start = step*BATCH
            targets = batch_targets(bank, start, start+BATCH)
            net.zero_grad(set_to_none=True)
            on, off, stats = base.backward(net, inputs, ids[start:start+BATCH], int(code), targets, mode, audit_routing=True)
            replay = float((on.cpu()-bank['targets']['clean'][start:start+BATCH]).abs().max())
            max_replay = max(max_replay, replay)
            assert replay < 1e-4, (mode, replay)
            assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
            stats['classification_and_aux_gradients'] = {
                g['name']: sum(float(p.grad.square().sum()) for p in g['params'] if p.grad is not None)**.5
                for g in groups}
            # Routing must not change backbone/other-module gradients before shared clipping.
            others = {n:p.grad.detach().cpu().clone() for n,p in net.named_parameters()
                      if p.grad is not None and not n.startswith(('ccrel.','exchange.'))}
            if mode == 'none':
                baseline_grads[step] = others
            else:
                error = max(float((g-baseline_grads[step][n]).abs().max()) for n,g in others.items())
                errors.append(error)
            active = net.exchange.trace['active'].any(1).detach()
            assert torch.equal(active.cpu(), bank['active'][start:start+BATCH])
            rows.extend(source_rows(bank,start,on,off,active,targets,0,mode))
            norms.append(stats)
        write_rows(out/f'verify_{mode}_local.csv', rows)
        reports[mode] = dict(batches=norms, repeated_backward_non_target_difference=max(errors,default=0.))
    t = bank['targets']
    matched = t['matched']
    assert bool(matched.any()), 'Real check did not exercise a donor pair'
    assert torch.equal(bank['labels'][matched],bank['labels'][t['donor'][matched]])
    assert bool((torch.where(matched)[0] != t['donor'][matched]).all())
    assert float(sum(row['auxiliary_norm']['exchange'] for row in reports['measured']['batches'])) > 0
    assert float(sum(row['auxiliary_norm']['reader'] for row in reports['measured']['batches'])) > 0
    for group in groups:
        assert sum(row['classification_and_aux_gradients'][group['name']]
                   for row in reports['none']['batches']) > 0, group['name']
    result = dict(passed=True, optimizer_steps=0, samples=16,
        selected=int(t['use'].sum()), eligible=int(t['eligible'].sum()), matched=int(matched.sum()),
        excluded_singletons=int((t['eligible'] & ~matched).sum()),
        replay_max=max_replay, modes=reports, seconds=bank['seconds'],
        max_cuda_gib=torch.cuda.max_memory_allocated()/1024**3,
        model_variant=variant, selection_mode=getattr(net,'selection_mode',None),
        optimizer_groups={g['name']:dict(lr=g['lr'],parameters=sum(p.numel() for p in g['params'])) for g in groups},
        formal_model_modified=False, external_evaluation=False)
    (out/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result), flush=True)


def main(default_model='legacy'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('verify','prepare','train'), default='verify')
    parser.add_argument('--mode', choices=MATCHED_MODES, default='measured')
    parser.add_argument('--bank', type=Path)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--steps', type=int, default=120)
    parser.add_argument('--model', choices=('legacy','unified'), default=default_model)
    args = parser.parse_args()
    if args.phase == 'verify':
        verify(args.model)
    elif args.phase == 'prepare':
        prepare(args.seed,args.steps,args.model)
    else:
        if args.bank is None:
            parser.error('--bank is required for train')
        train(args.bank,args.mode,args.model)


if __name__ == '__main__':
    main()
