"""Frozen M3 reference controls; preserve the original PCEA source statistics."""
from pathlib import Path
from collections import Counter
import argparse
import gc
import importlib.util
import json
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
PAPER = ROOT.parent
OLD = PAPER/'pcea_routed_correction_20261118'
STORE = Path('C:/Users/25791/AppData/Local/sar_ports24_fullscale_20261110')
sys.path.insert(0, str(ROOT))
from audit import read, save, passage


def load_model():
    spec = importlib.util.spec_from_file_location('reference_control_model', PAPER/'pcea_class_local_20261117/model.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def size_vector(row):
    return np.log([[row[p]['ship_axis'][k] for k in ('length_px', 'width_px')]
                   for p in ('vv', 'vh')]).flatten()


def choose_banks(m):
    original = read(STORE/'manifest.json')
    compact = read(PAPER/'m3_coarse_supplement_20261118/evaluation_inputs/manifest_local.json')
    eligible = read(ROOT/'eligible_local.json')
    query, refs = compact[:358], compact[358:]
    query_ports = {r['port'] for r in query}
    banks, source_ids, details = dict(original=refs), {}, []
    recovered = {}
    for family in ('same_port', 'outside_query_ports'):
        for draw in (1, 2, 3):
            rng = np.random.default_rng(2026111800+draw)
            bank, ids, used = [], [], set()
            for slot, ref in enumerate(refs):
                candidates = [i for i in eligible if original[i]['label'] == ref['label']
                    and original[i]['mmsi'] not in used
                    and (original[i]['port'] == ref['port'] if family == 'same_port'
                         else original[i]['port'] not in query_ports | {ref['port']})]
                # Match observable target size, never a classifier score or query label.
                ranked = []
                for i in candidates:
                    raw = original[i]
                    if not (raw.get('vv') and raw.get('vh')):
                        continue
                    d = float(np.square(size_vector(raw)-size_vector(ref)).mean())
                    ranked.append((d, i))
                ranked.sort()
                chosen = None
                for offset in range(0, len(ranked), 20):
                    pool = ranked[offset:offset+20]
                    for j in rng.permutation(len(pool)):
                        distance, i = pool[j]
                        row = recovered.setdefault(i, original[i])
                        views = m.old.t.e.task.data.prepare(row)
                        if all(v is not None and v['valid_target'] for v in views):
                            chosen = row, i, distance
                            break
                    if chosen is not None:
                        break
                if chosen is None:
                    raise RuntimeError(f'No exact eligible reference: {family}/{draw}/{slot}')
                row, idx, distance = chosen
                bank.append(row)
                ids.append(idx)
                used.add(row['mmsi'])
                details.append(dict(bank=f'{family}_{draw}', slot=slot, source_index=idx,
                    label=row['label'], port=row['port'], size_log_mse=distance))
            name = f'{family}_{draw}'
            banks[name], source_ids[name] = bank, ids
            assert Counter(r['label'] for r in bank) == Counter(r['label'] for r in refs)
            assert len(used) == 16
    save(ROOT/'banks_local.json', banks)
    save(ROOT/'selection_local.json', details)
    exposure = {}
    for seed in (17, 29, 43):
        dest = OLD/f'fits/routed_s{seed}'
        step = read(dest/'receipt.json')['best']['step']
        batch = read(dest/'config.json')['batch']
        seen = set(np.load(dest/'training_ids_local.npy')[:step*batch].tolist())
        exposure[str(seed)] = {name: sum(i in seen for i in ids) for name, ids in source_ids.items()}
    save(ROOT/'selection_summary.json', dict(
        banks={k: dict(n=len(v), classes=dict(Counter(r['label'] for r in v)),
            ports=dict(Counter(r['port'] for r in v)), unique_vessels=len({r['mmsi'] for r in v})) for k,v in banks.items()},
        continuation_training_exposure=exposure,
        size_log_mse={k: float(np.mean([r['size_log_mse'] for r in details if r['bank'] == k])) for k in source_ids},
        alternative_unique_vessels=len({r['mmsi'] for k,v in banks.items() if k != 'original' for r in v}),
        all_queries=358, unique_query_vessels=len({r['mmsi'] for r in query}),
        selection_uses_predictions=False))
    return query, banks


class Inputs:
    def __init__(self, rows, m):
        self.rows = rows
        shapes = dict(pixels=(2,160,160), valid=(2,160,160), available=(2,),
            slots=(2,20,10,10), local_slots=(2,20,20,20), read=(2,20,20,20),
            write=(2,20,10,10), mass=(2,20))
        self.arrays = {k: np.zeros((len(rows),)+shape, np.uint8 if k in ('pixels','valid','available')
                                  else np.float32) for k,shape in shapes.items()}
        for i,row in enumerate(rows):
            for p,view in enumerate(m.old.t.e.task.data.prepare(row)):
                if view is None or not view['valid_target']:
                    continue
                for k,v in (('pixels',view['pixels']),('valid',view['footprint']),('available',1)):
                    self.arrays[k][i,p] = v
                native = m.old.t.raw.tokens.slots(view['geometry'],view['footprint'])
                for k,v in m.old.t.geometry(native).items():
                    self.arrays[k][i,p] = v
        assert self.arrays['available'].any(1).all()

    def data(self, ids):
        out = {k:torch.as_tensor(v[ids],device='cuda').float() for k,v in self.arrays.items()}
        out['images'] = out.pop('pixels').unsqueeze(2)/255
        out['valid'] = out['valid'].unsqueeze(2)
        return out


def main():
    global np, torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    m = load_model()
    np,torch = m.np,m.torch
    torch.set_num_threads(2)
    query,banks = choose_banks(m)
    rows = list(query)
    bank_ids = {}
    for name,bank in banks.items():
        bank_ids[name] = np.arange(len(rows),len(rows)+len(bank))
        rows.extend(bank)
    inputs = Inputs(rows,m)
    assert inputs.arrays['available'][358:].all()
    budget = {}
    for name,indices in bank_ids.items():
        legal = m.old.t.e.task.legal(rows,np.arange(358),indices).cpu().numpy()
        counts = []
        for label in range(8):
            counts.append(legal[:,np.array([r['label'] for r in banks[name]]) == label].sum(1))
        assert np.array_equal(np.stack(counts,1),np.full((358,8),2)),name
        budget[name] = dict(per_available_polarization_per_class=2, all_query_reference_pairs_legal=True)
    save(ROOT/'effective_reference_budget.json',budget)
    save(ROOT/'manifest_local.json', rows)
    save(ROOT/'bank_indices_local.json', {k:v.tolist() for k,v in bank_ids.items()})
    y = np.array([r['label'] for r in query])
    results, checks = [], []
    started = time.perf_counter()
    for seed in ((17,) if args.smoke else (17,29,43)):
        net = m.Model('generic')
        net.load_state_dict(torch.load(OLD/f'fits/routed_s{seed}/best.pt',map_location='cuda',weights_only=True),strict=True)
        net.requires_grad_(False).eval()
        with torch.no_grad():
            refs = {'original':net.reference(inputs.data(bank_ids['original']))}
            source = net.exchange.source_raw.clone()
            weight = net.exchange.source_weight.clone()
            for name,ids in bank_ids.items():
                if name == 'original':
                    continue
                a,_ = net.extract(inputs.data(ids),True)
                refs[name] = {k:a[k] for k in ('desc','mass','available')}
            assert torch.equal(source,net.exchange.source_raw) and torch.equal(weight,net.exchange.source_weight)
            for split,all_ids in (('validation',np.arange(160)),('expansion',np.arange(160,358))):
                ids = all_ids[:12] if args.smoke else all_ids
                output = {k:[] for k in ('m3_off',*banks)}
                cache = np.load(OLD/f'fits/routed_s{seed}/{split}_on.npz')
                expected = read(STORE/'groups.json')['validation'] if split == 'validation' else list(range(198))
                assert np.array_equal(cache['ids'],expected)
                assert np.array_equal(cache['label'],y[all_ids])
                for first in range(0,len(ids),12):
                    part = ids[first:first+12]
                    a,c = net.extract(inputs.data(part),True)
                    qrows = [query[i] for i in part]
                    zoff,_ = net.finish(a,c,qrows,refs['original'],banks['original'],m3_on=False)
                    output['m3_off'].append(zoff.cpu().numpy())
                    for name in banks:
                        z,_ = net.finish(a,c,qrows,refs[name],banks[name])
                        output[name].append(z.cpu().numpy())
                        if first == 0:
                            otheroff,_ = net.finish(a,c,qrows,refs[name],banks[name],m3_on=False)
                            error = float((zoff-otheroff).abs().max())
                            assert error < 1e-7,(name,error)
                            checks.append(dict(seed=seed,split=split,bank=name,m3_off_replay_error=error))
                    if first % 60 == 0:
                        print(seed,split,first+len(part),len(ids),round(time.perf_counter()-started,1),flush=True)
                scores = {k:np.concatenate(v) for k,v in output.items()}
                error = float(abs(scores['original']-cache['score'][:len(ids)]).max())
                assert error < 2e-5,(seed,split,error)
                checks.append(dict(seed=seed,split=split,original_replay_error=error))
                before = scores['m3_off'].argmax(1) == y[ids]
                for name,z in scores.items():
                    after = z.argmax(1) == y[ids]
                    row = dict(seed=seed,split=split,bank=name,**m.full.metrics(y[ids],z),
                        rescued=int((~before & after).sum()),harmed=int((before & ~after).sum()))
                    results.append(row)
                    np.savez_compressed(ROOT/f'{"smoke_" if args.smoke else ""}{split}_s{seed}_{name}_local.npz',
                                        ids=ids,label=y[ids],score=z)
                print('RESULT',seed,split,[(r['bank'],round(r['oa'],3)) for r in results[-len(scores):]],flush=True)
                save(ROOT/('smoke_results.json' if args.smoke else 'results.json'),results)
                save(ROOT/('smoke_checks.json' if args.smoke else 'checks.json'),checks)
        del net
        gc.collect()
        torch.cuda.empty_cache()
    save(ROOT/('smoke_status.json' if args.smoke else 'status.json'),dict(complete=True,
        seconds=time.perf_counter()-started,training_steps=0,test3075_read=False,
        pcea_statistics='original16 fixed',seeds=[17] if args.smoke else [17,29,43]))


if __name__ == '__main__':
    main()
