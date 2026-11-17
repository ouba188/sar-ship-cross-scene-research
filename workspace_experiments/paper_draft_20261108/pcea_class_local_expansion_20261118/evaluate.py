"""Frozen nine-checkpoint evaluation on new vessels within known passages."""
from pathlib import Path
import csv
import gc
import importlib.util
import json
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent/'pcea_class_local_20261117'
WORK = ROOT.parents[1]
STORE = Path('C:/Users/25791/AppData/Local/sar_ports24_fullscale_20261110')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


m = load('expansion_frozen_model', OLD/'model.py')
prep = load('expansion_manifest_helper', ROOT.parent/'ports24_fullscale_20261110/prepare_manifest.py')
np, torch = m.np, m.torch
t = m.old.t


class Inputs:
    def __init__(self):
        excluded = set(read(ROOT/'exposure_review.json')['historical_training_corpus_rows'])
        candidates = [r for r in read(ROOT/'candidates_local.json') if r['corpus_global_row'] not in excluded]
        wanted = {str(r['corpus_global_row']) for r in candidates}
        views = {(r['global_row'], r['pol']): r for r in prep.read_csv(WORK/'chip_quality_20261017/per_view_quality.csv')
                 if r['global_row'] in wanted}
        dets = {}
        for r in prep.read_csv(WORK/'chip_quality_20261017/detected_objects.csv'):
            if r['global_row'] not in wanted:
                continue
            dets.setdefault((r['global_row'], r['pol']), {})[int(r['detection_index'])] = dict(
                xywhr=json.loads(r['xywhr']), corners=json.loads(r['corners']),
                confidence=float(r['confidence']), shape_usable=r['shape_usable']=='True',
                center_inside=r['center_inside']=='True', touches_edge=r['touches_edge']=='True')
        self.rows = []
        for r in candidates:
            index = str(r['corpus_global_row'])
            parts = r['product'].split('_')
            row = dict(id='full_'+index, corpus_global_row=int(index), label=int(r['class_id']),
                       mmsi=r['mmsi'].removesuffix('.0'), product=r['product'], port=r['port'],
                       pass_group='_'.join((parts[0], parts[6], parts[7])))
            for p in ('vv', 'vh'):
                row[p] = prep.make_view(views.get((index, p)), dets.get((index, p), {}))
            assert row['vv'] is not None or row['vh'] is not None
            self.rows.append(row)
        n = len(self.rows)
        original, groups = read(STORE/'manifest.json'), read(STORE/'groups.json')
        known = groups['validation'][:8]
        for i in groups['references']+known:
            fixed, _ = m.old.recovered.recovery.recover(original[i])
            self.rows.append(fixed)
        self.groups = dict(validation=np.arange(n), references=np.arange(n, n+16), replay=np.arange(n+16, n+24))
        self.reference_rows = self.rows[n:n+16]
        self.y = np.array([r['label'] for r in self.rows])
        shapes = dict(pixels=(2,160,160), valid=(2,160,160), available=(2,),
                      slots=(2,20,10,10), local_slots=(2,20,20,20), read=(2,20,20,20),
                      write=(2,20,10,10), mass=(2,20))
        self.arrays = {k:np.zeros((len(self.rows),)+shape, dtype=np.uint8 if k in ('pixels','valid','available') else np.float32)
                       for k,shape in shapes.items()}
        for i, row in enumerate(self.rows):
            for p, view in enumerate(t.e.task.data.prepare(row)):
                if view is None or not view['valid_target']:
                    continue
                self.arrays['pixels'][i,p] = view['pixels']
                self.arrays['valid'][i,p] = view['footprint']
                self.arrays['available'][i,p] = 1
                native = t.raw.tokens.slots(view['geometry'], view['footprint'])
                for key, val in t.geometry(native).items():
                    self.arrays[key][i,p] = val
        assert self.arrays['available'][:n].any(1).all()
        write(ROOT/'evaluation_manifest_local.json', self.rows)
        np.savez_compressed(ROOT/'native_local.npz', pixels=self.arrays['pixels'], available=self.arrays['available'])
        write(ROOT/'input_scope.json', dict(n=n, classes=np.bincount(self.y[:n],minlength=8).tolist(),
              ports={p:sum(r['port']==p for r in self.rows[:n]) for p in sorted({r['port'] for r in self.rows[:n]})},
              dual=int(self.arrays['available'][:n].all(1).sum()),
              single=int((self.arrays['available'][:n].sum(1)==1).sum()),
              reference_dual=int(self.arrays['available'][n:n+16].all(1).sum()),
              crop='existing native160 crop/pad; no resize',
              recovery='original16 recovered references retained; new queries use existing accepted detections; no new recovery index',
              independent_passage=False, test_images_read=False))

    def data(self, ids, split=None, code=0):
        assert code == 0
        data = {k:torch.as_tensor(v[ids].copy(), device='cuda').float() for k,v in self.arrays.items()}
        data['images'] = data.pop('pixels').unsqueeze(2)/255
        data['valid'] = data['valid'].unsqueeze(2)
        return data


def main():
    assert read(ROOT/'exposure_review.json')['eligible_for_known_passage_evaluation']
    assert all((OLD/'fits'/f'{a}_s{s}/receipt.json').is_file() for a in m.ARMS for s in (17,29,43))
    inputs = Inputs()
    ids = inputs.groups['validation']
    results = []
    for seed in (17,29,43):
        for arm in m.ARMS:
            net = m.Model(arm)
            net.load_state_dict(torch.load(OLD/'fits'/f'{arm}_s{seed}/best.pt', map_location='cuda', weights_only=True), strict=True)
            net.requires_grad_(False)
            net.accelerate()
            if seed == 17 and arm == 'full':
                z = m.full.predict(net, inputs, inputs.groups['replay'], 'validation')
                old = np.load(OLD/'fits/full_s17/validation_on.npz')['score'][:8]
                error = float(abs(z-old).max())
                assert error < 1e-4 and np.array_equal(z.argmax(1), old.argmax(1)), error
                write(ROOT/'replay.json', dict(max_error=error, predictions_equal=True, n=8))
            modes = ['on'] if arm == 'full' else ['on','off']
            if arm == 'class_detail':
                modes += ['uniform','permuted']
            for mode in modes:
                if arm != 'full':
                    net.exchange.disabled = mode == 'off'
                    net.exchange.uniform_detail = mode == 'uniform'
                    net.exchange.permute_detail = mode == 'permuted'
                z = m.full.predict(net, inputs, ids, 'validation')
                np.savez_compressed(ROOT/f'{arm}_s{seed}_{mode}_local.npz', ids=ids, label=inputs.y[ids], score=z)
                results.append(dict(arm=arm, seed=seed, mode=mode, **m.full.metrics(inputs.y[ids], z)))
                write(ROOT/'results.json', results)
                print(arm, seed, mode, results[-1]['oa'], results[-1]['ba'], flush=True)
            del net
            gc.collect()
            torch.cuda.empty_cache()
    write(ROOT/'status.json', dict(complete=True, trained=False, forwards=len(results),
          test_images_read=False, test_predictions_computed=False))


if __name__ == '__main__':
    main()
