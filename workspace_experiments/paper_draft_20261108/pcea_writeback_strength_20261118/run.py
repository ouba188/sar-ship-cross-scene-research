"""Scale actual frozen PCEA feature updates; select only on source-inner data."""
from pathlib import Path
import argparse
import gc
import importlib.util
import json
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent/'pcea_class_local_20261117'
EXPANSION = ROOT.parent/'pcea_class_local_expansion_20261118'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


m = load('strength_previous_model',OLD/'model.py')
np, torch = m.np, m.torch
SEEDS = (17,29,43)
SCALES = (0,1,2,4,8)


class Strength:
    def __init__(self, net):
        self.scale = 1
        self.trace = []
        self.handle = net.exchange.register_forward_hook(self.forward)

    def forward(self, module, args, output):
        x = args[0]
        delta = module.trace['delta']
        self.trace.append(torch.stack((x.square().mean((2,3,4)).sqrt(),
                                       delta.square().mean((2,3,4)).sqrt()),-1).detach().cpu().numpy())
        if self.scale == 1:
            return output
        if self.scale == 0:
            return x
        return x+self.scale*delta

    def reset(self, scale):
        self.scale, self.trace = scale, []

    def stats(self):
        value = np.concatenate(self.trace)
        ratio = self.scale*value[...,1]/np.maximum(value[...,0],1e-12)
        return dict(update_input_rms_median=float(np.median(ratio)),
                    update_input_rms_p95=float(np.quantile(ratio,.95)),
                    update_input_rms_max=float(ratio.max()))


def model(seed):
    net = m.Model('generic')
    net.load_state_dict(torch.load(OLD/'fits'/f'generic_s{seed}/best.pt',map_location='cuda',weights_only=True),strict=True)
    net.requires_grad_(False)
    net.accelerate()
    return net


def smoke(inputs, inner):
    net = model(17)
    ids = inner[:12]
    original = m.full.predict(net,inputs,ids,'small')
    saved = np.load(OLD/'fits/generic_s17/inner.npz')
    assert np.array_equal(saved['ids'][:12],ids)
    replay = float(abs(original-saved['score'][:12]).max())
    assert replay < 1e-4
    net.exchange.disabled = True
    off = m.full.predict(net,inputs,ids,'small')
    net.exchange.disabled = False
    strength = Strength(net)
    strength.reset(0)
    zero = m.full.predict(net,inputs,ids,'small')
    strength.reset(1)
    one = m.full.predict(net,inputs,ids,'small')
    one_stats = strength.stats()
    strength.reset(8)
    eight = m.full.predict(net,inputs,ids,'small')
    eight_stats = strength.stats()
    assert float(abs(zero-off).max()) < 1e-6
    assert float(abs(one-original).max()) < 1e-6
    assert np.isfinite(eight).all() and float(abs(eight-one).max()) > 1e-5
    assert abs(eight_stats['update_input_rms_median']-8*one_stats['update_input_rms_median']) < 1e-6
    write(ROOT/'smoke.json',dict(passed=True,old_replay_max=replay,
          zero_vs_disabled_max=float(abs(zero-off).max()),one_vs_original_max=float(abs(one-original).max()),
          eight_vs_one_logit_max=float(abs(eight-one).max()),one=one_stats,eight=eight_stats))
    strength.handle.remove()
    del net
    gc.collect()
    torch.cuda.empty_cache()


def source_scan(inputs, inner):
    records = []
    for seed in SEEDS:
        net = model(seed)
        strength = Strength(net)
        for scale in SCALES:
            strength.reset(scale)
            start = time.perf_counter()
            z = m.full.predict(net,inputs,inner,'small')
            assert np.isfinite(z).all()
            np.savez_compressed(ROOT/f'inner_s{seed}_a{scale}_local.npz',ids=inner,label=inputs.y[inner],score=z)
            records.append(dict(seed=seed,scale=scale,seconds=time.perf_counter()-start,
                                **m.full.metrics(inputs.y[inner],z),**strength.stats()))
            write(ROOT/'source_metrics.json',records)
            print('SOURCE',seed,scale,records[-1]['oa'],records[-1]['ba'],flush=True)
        strength.handle.remove()
        del net
        gc.collect()
        torch.cuda.empty_cache()
    means = {str(scale):{k:float(np.mean([r[k] for r in records if r['scale']==scale]))
                       for k in ('oa','ba','macro_f1','nll')} for scale in SCALES}
    selected = max(SCALES,key=lambda a:(round(means[str(a)]['ba'],10),round(means[str(a)]['oa'],10),-a))
    write(ROOT/'selection.json',dict(scale=selected,means=means,
          rule='common scale by source mean BA/OA, then smaller scale; no external selection',
          source_seen_by_earlier_encoder=True))
    print('SELECTED',selected,flush=True)
    return selected


def external(inputs, selected):
    # Only the source-selected strength is forwarded outside the source-inner pool.
    expansion_module = load('strength_expansion_inputs',EXPANSION/'evaluate.py')
    extra = expansion_module.Inputs()
    records = []
    for seed in SEEDS:
        net = model(seed)
        strength = Strength(net)
        for name, data, ids in [('validation160',inputs,inputs.groups['validation']),
                                 ('expansion198',extra,extra.groups['validation'])]:
            strength.reset(selected)
            z = m.full.predict(net,data,ids,'validation')
            assert np.isfinite(z).all()
            np.savez_compressed(ROOT/f'{name}_s{seed}_selected_local.npz',ids=ids,label=data.y[ids],score=z)
            records.append(dict(split=name,seed=seed,scale=selected,**m.full.metrics(data.y[ids],z),**strength.stats()))
            write(ROOT/'external_metrics.json',records)
            print('EXTERNAL',name,seed,selected,records[-1]['oa'],records[-1]['ba'],flush=True)
        strength.handle.remove()
        del net
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase',choices=('smoke','run'),default='smoke')
    args = parser.parse_args()
    inputs = m.Inputs()
    inner = np.array(read(OLD/'split_local.json')['inner'],dtype=np.int64)
    write(ROOT/'scope.json',dict(source_inner=read(OLD/'scope.json')['inner'],
          checkpoint='generic previous best.pt seeds17/29/43; all initially formal17',
          scales=SCALES,no_training=True,test_images_read=False))
    if args.phase == 'smoke':
        smoke(inputs,inner)
    else:
        assert read(ROOT/'smoke.json')['passed']
        if (ROOT/'selection.json').exists():
            raise RuntimeError('This scan is already selected; inspect rather than overwrite.')
        chosen = source_scan(inputs,inner)
        external(inputs,chosen)
        write(ROOT/'status.json',dict(complete=True,trained=False,source_conditions=15,external_conditions=6,test_images_read=False))
