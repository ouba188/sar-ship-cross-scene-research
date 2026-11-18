"""Frozen M3 slot-granularity intervention; no optimizer or new model weights."""
from pathlib import Path
import argparse
import gc
import importlib.util
import json
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
PAPER = ROOT.parent
OLD = PAPER / 'pcea_routed_correction_20261118'
EXTRA = PAPER / 'pcea_class_local_expansion_20261118'
STORE = Path('C:/Users/25791/AppData/Local/sar_ports24_fullscale_20261110')
MODES = ('fixed', 'coarse', 'adaptive')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def groups_for(length, width, mode):
    if mode == 'fixed':
        return [[i] for i in range(20)]
    if mode == 'coarse':
        longitudinal, transverse = 2, 1
    else:
        longitudinal = max(n for n in (1, 2, 4) if n == 1 or length >= 16*n)
        transverse = 2 if width >= 32 else 1
    groups = []
    for offset in (0, 8):
        for a in range(longitudinal):
            for b in range(transverse):
                groups.append([offset+2*x+y for x in range(a*4//longitudinal, (a+1)*4//longitudinal)
                               for y in range(b*2//transverse, (b+1)*2//transverse)])
    for a in range(2):
        for b in range(transverse):
            groups.append([16+2*a+y for y in range(b*2//transverse, (b+1)*2//transverse)])
    return groups


def merge_matrix(rows, mode):
    matrix = np.zeros((len(rows), 2, 20, 20), np.float32)
    records = []
    for j, row in enumerate(rows):
        for p, pol in enumerate(('vv', 'vh')):
            view = row.get(pol)
            if view is None:
                continue
            axis = view['ship_axis']
            length, width = axis['length_px'], axis['width_px']
            groups = groups_for(length, width, mode)
            occupied = {0: 0, 8: 0, 16: 0}
            for group in groups:
                start = 0 if group[0] < 8 else 8 if group[0] < 16 else 16
                dest = start+occupied[start]
                matrix[j, p, dest, group] = 1
                occupied[start] += 1
            assert sorted(i for group in groups for i in group) == list(range(20))
            records.append(dict(id=row['id'], polarization=pol, length=float(length), width=float(width),
                                nominal_slots=len(groups), body_slots=occupied[0], end_slots=occupied[16]))
    return torch.as_tensor(matrix, device='cuda'), records


def merge(a, matrix):
    mass = torch.einsum('bpij,bpj->bpi', matrix, a['mass'])
    desc = torch.einsum('bpij,bpjd->bpid', matrix, a['desc']*a['mass'][..., None])
    desc = desc/mass[..., None].clamp_min(1e-20)
    out = dict(a, mass=mass, desc=desc)
    if 'write' in a:
        out['write'] = torch.einsum('bpij,bpjhw->bpihw', matrix, a['write'])
    return out


class Inputs:
    def __init__(self):
        original, groups = read(STORE/'manifest.json'), read(STORE/'groups.json')
        expansion = read(EXTRA/'evaluation_manifest_local.json')
        validation = [m.old.recovered.recovery.recover(original[i])[0] for i in groups['validation']]
        self.rows = validation + expansion[:198] + expansion[198:214]
        self.splits = dict(validation=np.arange(160), expansion=np.arange(160, 358))
        self.refs = np.arange(358, 374)
        self.old_ids = dict(validation=np.asarray(groups['validation']), expansion=np.arange(198))
        assert [r['id'] for r in self.rows[358:]] == [original[i]['id'] for i in groups['references']]
        shapes = dict(pixels=(2, 160, 160), valid=(2, 160, 160), available=(2,),
                      slots=(2, 20, 10, 10), local_slots=(2, 20, 20, 20), read=(2, 20, 20, 20),
                      write=(2, 20, 10, 10), mass=(2, 20))
        self.arrays = {k: np.zeros((374,)+shape, np.uint8 if k in ('pixels', 'valid', 'available') else np.float32)
                       for k, shape in shapes.items()}
        for i, row in enumerate(self.rows):
            for p, view in enumerate(m.old.t.e.task.data.prepare(row)):
                if view is None or not view['valid_target']:
                    continue
                self.arrays['pixels'][i, p] = view['pixels']
                self.arrays['valid'][i, p] = view['footprint']
                self.arrays['available'][i, p] = 1
                native = m.old.t.raw.tokens.slots(view['geometry'], view['footprint'])
                for key, value in m.old.t.geometry(native).items():
                    self.arrays[key][i, p] = value
        self.y = np.array([r['label'] for r in self.rows])
        self.matrices, self.records = {}, {}
        for mode in MODES:
            self.matrices[mode], self.records[mode] = merge_matrix(self.rows, mode)
        assert self.arrays['available'].any(1).all()
        save('manifest_local.json', self.rows)
        save('geometry_local.json', self.records)

    def data(self, ids):
        result = {k: torch.as_tensor(v[ids], device='cuda').float() for k, v in self.arrays.items()}
        result['images'] = result.pop('pixels').unsqueeze(2)/255
        result['valid'] = result['valid'].unsqueeze(2)
        return result


def verify_merge(a, merged, matrix):
    errors = {}
    for start, end in ((0, 8), (8, 16), (16, 20)):
        errors[f'mass_{start}'] = float((a['mass'][:, :, start:end].sum(2)-merged['mass'][:, :, start:end].sum(2)).abs().max())
        errors[f'write_{start}'] = float((a['write'][:, :, start:end].sum(2)-merged['write'][:, :, start:end].sum(2)).abs().max())
    assert max(errors.values()) < 1e-4, errors
    assert torch.isfinite(merged['desc']).all()
    assert (merged['desc'][merged['mass'] == 0] == 0).all()
    return errors


def main():
    global m, torch, np
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--side-probe', action='store_true')
    args = parser.parse_args()
    modes = ('fixed', 'adaptive_query', 'adaptive_reference') if args.side_probe else MODES
    prefix = 'side_' if args.side_probe else ''
    m = load('adaptive_slot_model', PAPER/'pcea_class_local_20261117/model.py')
    torch, np = m.torch, m.np
    torch.set_num_threads(2)
    print('Preparing only 160+198 queries and original16 references', flush=True)
    inputs = Inputs()
    records, checks = [], []
    tick = time.perf_counter()
    for seed in ((43,) if args.side_probe else (17,) if args.smoke else (17, 29, 43)):
        net = m.Model('generic')
        net.load_state_dict(torch.load(OLD/f'fits/routed_s{seed}/best.pt', map_location='cuda', weights_only=True), strict=True)
        net.requires_grad_(False).eval()
        net.accelerate()
        with torch.no_grad():
            ref = net.reference(inputs.data(inputs.refs))
            refs = {mode: ref if mode in ('fixed', 'adaptive_query') else merge(
                ref, inputs.matrices['adaptive' if mode == 'adaptive_reference' else mode][inputs.refs]) for mode in modes}
            for split, all_ids in inputs.splits.items():
                ids = all_ids[:12] if args.smoke else all_ids
                outputs = {mode: [] for mode in modes}
                old_cache = np.load(OLD/f'fits/routed_s{seed}/{split}_on.npz')
                assert np.array_equal(old_cache['ids'], inputs.old_ids[split])
                for first in range(0, len(ids), 12):
                    part = ids[first:first+12]
                    data = inputs.data(part)
                    a, c = net.extract(data, True)
                    for mode in modes:
                        aa = a if mode in ('fixed', 'adaptive_reference') else merge(
                            a, inputs.matrices['adaptive' if mode == 'adaptive_query' else mode][part])
                        if first == 0 and seed == 17:
                            record = verify_merge(a, aa, inputs.matrices[mode][part])
                            if mode != 'fixed':
                                readout = torch.einsum('bpij,bpjhw->bpihw', inputs.matrices[mode][part], data['read'])
                                merged_data = dict(data, read=readout)
                                direct, _ = net.extract(merged_data, True)
                                record['direct_pool'] = float((aa['desc']-direct['desc']).abs().max())
                                assert record['direct_pool'] < 2e-5, record
                            checks.append(dict(split=split, mode=mode, **record))
                        score, _ = net.finish(aa, c, [inputs.rows[i] for i in part], refs[mode], inputs.rows[358:])
                        outputs[mode].append(score.cpu().numpy())
                    if first % 60 == 0:
                        print(seed, split, first+len(part), '/', len(ids), round(time.perf_counter()-tick, 1), flush=True)
                scores = {mode: np.concatenate(value) for mode, value in outputs.items()}
                error = float(abs(scores['fixed']-old_cache['score'][:len(ids)]).max())
                assert error < 2e-5, (seed, split, error)
                checks.append(dict(seed=seed, split=split, fixed_replay_error=error))
                y = inputs.y[ids]
                for mode, score in scores.items():
                    name = f'{prefix}{split}_s{seed}_{mode}'
                    np.savez_compressed(ROOT/f'{name}_local.npz', ids=ids, label=y, score=score)
                    before, after = scores['fixed'].argmax(1) == y, score.argmax(1) == y
                    row = dict(seed=seed, split=split, mode=mode, n=len(ids), **m.full.metrics(y, score),
                               rescued=int((~before & after).sum()), harmed=int((before & ~after).sum()))
                    records.append(row)
                    print(name, row['oa'], row['ba'], row['rescued'], row['harmed'], flush=True)
                save('smoke_results.json' if args.smoke else prefix+'results.json', records)
                save(prefix+'checks.json', checks)
        del net
        gc.collect()
        torch.cuda.empty_cache()
    save('smoke_status.json' if args.smoke else prefix+'status.json', dict(complete=True, trained=False, smoke=args.smoke,
         seconds=time.perf_counter()-tick, evaluations=len(records), test3075_read=False))


if __name__ == '__main__':
    main()
