"""Matched intervention selection controls; no formal model changes."""
from pathlib import Path
import argparse
import gc
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent/'staged_ablation_20261109'
sys.path.insert(0, str(SOURCE))
import staged as e
import intervention_inputs
from sar_m123.current import training

torch = e.torch
SEEDS = (17,29,43)
ARMS = ('random_selection','no_protection')
COEFFICIENT = .3683066666126251
ORIGINAL_LOSS = training.adaptation_loss


def targets(seed, arm, rows, ids):
    source = SOURCE/f'full_s{seed}_lr1/intervention/teacher.json'
    items = json.loads(source.read_text(encoding='utf-8'))
    assert [r['index'] for r in items] == ids.tolist()
    assert all(r['id']==rows[r['index']]['id'] and r['label']==rows[r['index']]['label'] for r in items)
    original = np.array([r['useful'] for r in items], dtype=bool)
    labels = np.array([r['label'] for r in items])
    flags = original.copy()
    if arm == 'random_selection':
        rng = np.random.default_rng(20261111+seed)
        for label in range(4):
            chosen = np.flatnonzero(labels==label)
            flags[chosen] = rng.permutation(original[chosen])
            assert flags[chosen].sum() == original[chosen].sum()
        assert np.any(flags != original)
    for j,r in enumerate(items):
        r['intervention_useful'] = bool(original[j])
        r['useful'] = bool(flags[j])
        r['protection_active'] = bool(flags[j]) and arm != 'no_protection'
        r['protection_rule'] = arm
    return items


def load_start(seed, protocol):
    # The complete carrier state replaces every tensor; no external pretrain file is needed.
    net = e.Network('full',seed,None)
    net.load_state_dict(torch.load(SOURCE/f'full_s{seed}_lr1/carrier/final.pt',map_location='cpu',weights_only=True))
    e.configure(net,-1)
    return net


def smoke(protocol, rows, groups, a, c):
    result = {}
    ids = groups['source_adapt']
    for seed in SEEDS:
        net = load_start(seed,protocol)
        original = targets(seed,'full',rows,ids)
        teacher = np.array([r['original'] for r in original],np.float32)
        score = e.predict(net,a,c,rows,ids,groups['references'])
        error = float(abs(score-teacher).max())
        assert error < 1e-4, (seed,error)
        random = targets(seed,'random_selection',rows,ids)
        y = torch.tensor([r['label'] for r in original],device=e.task.DEVICE)
        t = torch.tensor(teacher,device=e.task.DEVICE)
        # Force margin violations to test that disabling protection actually changes the loss.
        student = t.clone().scatter_add(1,y[:,None],t.new_full((len(ids),1),-.7)).requires_grad_(True)
        count = torch.bincount(y,minlength=4)
        w = len(ids)/(4*count[y]).float()
        u = torch.tensor([r['useful'] for r in original],device=e.task.DEVICE)
        loss,terms = ORIGINAL_LOSS(student,y,t,w,u,COEFFICIENT)
        off,off_terms = ORIGINAL_LOSS(student,y,t,w,u,0.)
        rand,rand_terms = ORIGINAL_LOSS(student,y,t,w,torch.tensor([r['useful'] for r in random],device=e.task.DEVICE),COEFFICIENT)
        assert torch.allclose(off,terms['ce']+.1*terms['keep'])
        assert float((loss-off).detach())>0
        assert torch.allclose(terms['ce'],rand_terms['ce']) and torch.allclose(terms['keep'],rand_terms['keep'])
        (loss+off+rand).backward()
        assert torch.isfinite(student.grad).all()
        result[str(seed)] = dict(teacher_replay_error=error,selected=int(u.sum()),random_changed=sum(r['useful']!=r['intervention_useful'] for r in random),
            protection_loss_delta=float((loss-off).detach()),loss_semantics_passed=True)
        del net,student,loss,off,rand
        gc.collect();torch.cuda.empty_cache()
    e.task.write(ROOT/'smoke.json',result)
    print('SMOKE_OK',result,flush=True)


def fit(arm,seed,protocol,rows,groups,a,c):
    folder = ROOT/f'{arm}_s{seed}'
    if (folder/'complete.json').exists(): return
    folder.mkdir(parents=True,exist_ok=True)
    net = load_start(seed,protocol)
    selected = targets(seed,arm,rows,groups['source_adapt'])
    e.task.write(folder/'selection.json',selected)
    parent = SOURCE/f'full_s{seed}_lr1'
    saved_select = intervention_inputs.select

    def cached_select(net,a,c,rows,ids,rid):
        assert ids.tolist()==[r['index'] for r in selected]
        return selected

    def controlled_loss(score,labels,teacher,weights,useful,coefficient):
        assert coefficient == COEFFICIENT
        return ORIGINAL_LOSS(score,labels,teacher,weights,useful,0. if arm=='no_protection' else coefficient)

    try:
        intervention_inputs.select = cached_select
        training.adaptation_loss = controlled_loss
        for cfg in protocol['phases'][1:]:
            dest = folder/cfg['name']
            e.task.write(ROOT/'status.json',dict(arm=arm,seed=seed,phase=cfg['name'],complete=False))
            if (dest/'receipt.json').exists():
                net.load_state_dict(torch.load(dest/'final.pt',map_location='cpu',weights_only=True))
                e.configure(net,cfg['phase'])
            else:
                e.train_phase(net,cfg,seed,folder,rows,groups,a,c,scale=1.)
            assert json.loads((dest/'orders.json').read_text()) == json.loads((parent/cfg['name']/'orders.json').read_text())
            if cfg['phase']==1: e.freeze_delta(net,a,rows,groups['references'])
    finally:
        intervention_inputs.select = saved_select
        training.adaptation_loss = ORIGINAL_LOSS
    e.configure(net,3);net.requires_grad_(False)
    metrics = {}
    for split in ('validation','evaluation'):
        ids = groups[split]
        score = e.predict(net,a,c,rows,ids,groups['references'])
        np.savez_compressed(folder/f'{split}.npz',ids=ids,score=score)
        metrics[split] = e.task.metrics(np.array([rows[i]['label'] for i in ids]),score)
    e.task.write(folder/'complete.json',dict(arm=arm,seed=seed,metrics=metrics,orders_identical=True,
        carrier_reused=True,coefficient=0. if arm=='no_protection' else COEFFICIENT,
        selected_count=sum(r['useful'] for r in selected),protected_count=sum(r['protection_active'] for r in selected),formal_model_changed=False))
    print('COMPLETE',arm,seed,metrics,flush=True)
    del net
    gc.collect();torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-only',action='store_true')
    args = parser.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    protocol = json.loads((SOURCE/'protocol.json').read_text(encoding='utf-8'))
    e.task.current.setup(e.task.DEVICE)
    rows,groups = e.prev.records();a,c = e.prev.arrays()
    a['write'] = np.load(e.PREVIOUS/'common_init/write.npy',mmap_mode='r')
    e.task.write(ROOT/'protocol.json',dict(arms=ARMS,seeds=SEEDS,parent=protocol,data={k:len(v) for k,v in groups.items()},
        start='matching Full carrier checkpoint',random='within-class permutation of useful, RNG 20261111+seed',
        coefficient=COEFFICIENT,teacher='fixed original Full teacher and original margins; no altered target',
        all_adapt_rows_receive_ce=True,suffix='intervention-reference-feedback-readout',
        source_hard_weights='recomputed by same original rule',selection='no new tuning; reused development test'))
    e.resident(a,c)
    smoke(protocol,rows,groups,a,c)
    if args.smoke_only:return
    for seed in SEEDS:
        for arm in ARMS: fit(arm,seed,protocol,rows,groups,a,c)
    e.task.write(ROOT/'status.json',dict(complete=True,arms=ARMS,seeds=SEEDS))


if __name__=='__main__':main()
