"""Actual R18 layer4 updates using the same cached prefix in every arm."""

from experiment import *

import copy
from concurrent.futures import ThreadPoolExecutor
import torch
from torch.nn import functional as F
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image

torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
CACHE = OUT/'cache'
STEPS = 96


def network():
    state = torch.load(WS/'e206_out/ssl_resnet18.pt', map_location='cpu', weights_only=True)
    net = resnet18(weights=None)
    missing, extra = net.load_state_dict(state['backbone'], strict=False)
    if set(missing) != {'fc.weight', 'fc.bias'} or extra:
        raise ValueError('Unexpected initial backbone')
    return net.cuda().eval()


def prepare_cache():
    CACHE.mkdir(exist_ok=True)
    net = network()
    prefix = torch.nn.Sequential(*list(net.children())[:-3])
    frame = pd.read_csv(PUBLIC/'target_manifest.csv')
    tf = transforms.Compose([transforms.Resize(128), transforms.ToTensor()])
    output = np.zeros((len(frame), 256, 8, 8), np.float16)
    nonstandard = {}
    feature = np.empty((len(frame), 512), np.float32)

    def read(path):
        with Image.open(path) as img:
            return tf(img.convert('RGB'))

    with torch.inference_mode(), ThreadPoolExecutor(max_workers=4) as pool:
        for lo in range(0, len(frame), 64):
            batch = list(pool.map(read, frame.chip_path_vh.iloc[lo:lo+64]))
            groups = {}
            for i, pixels in enumerate(batch):
                groups.setdefault(tuple(pixels.shape), []).append(i)
            for group in groups.values():
                value = prefix(torch.stack([batch[i] for i in group]).cuda()).half()
                rows = lo+np.asarray(group)
                if tuple(value.shape[1:]) == (256, 8, 8):
                    output[rows] = value.cpu().numpy()
                else:
                    for i, row in enumerate(rows):
                        nonstandard[str(int(row)+N_SOURCE)] = value[i].cpu().numpy()
                feature[rows] = net.layer4(value.float()).mean((2, 3)).cpu().numpy()
            if lo % 640 == 0:
                print('PREFIX', lo, '/', len(frame), flush=True)
        source = np.load(ROOT/'neural_complement_pilot_20261007/source_layer3.npy', mmap_mode='r')
        rows = np.array([0, 51, 213, 1845, 6237, 11828, 25922, 37315])
        fresh = net.layer4(torch.as_tensor(np.array(source[rows]), device='cuda').float()).mean((2, 3)).cpu().numpy()
    expected = np.load(ROOT/'neural_complement_pilot_20261007/frozen_replay/source_extra.npy', mmap_mode='r')[rows]
    np.testing.assert_allclose(fresh, expected, atol=2e-5, rtol=1e-5)
    np.save(CACHE/'public_prefix.npy', output)
    np.savez(CACHE/'nonstandard_prefix.npz', **nonstandard)
    np.save(CACHE/'public_initial_extra.npy', feature)
    with np.load(PUBLIC/'features.npz') as old:
        error = np.linalg.norm(feature-old['vh_ssl'], axis=1)/np.maximum(np.linalg.norm(old['vh_ssl'], axis=1), 1e-9)
    write(CACHE/'REPLAY.json', dict(source_max_abs=float(np.max(np.abs(fresh-expected))),
        public_old_cache_max_relative=float(error.max()), nonstandard_rows=list(nonstandard),
        frozen_and_updated_use_same_float16_prefix=True,
        initial_unreported_frozen_cache_run_replaced_before_evaluation=True))


class PrefixBank:
    def __init__(self):
        source = np.load(ROOT/'neural_complement_pilot_20261007/source_layer3.npy', mmap_mode='r')
        public = np.load(CACHE/'public_prefix.npy', mmap_mode='r')
        self.standard = torch.as_tensor(np.concatenate([source, public]), device='cuda')
        with np.load(CACHE/'nonstandard_prefix.npz') as z:
            self.other = {int(k): torch.as_tensor(z[k], device='cuda') for k in z.files}

    def apply(self, tail, rows):
        rows = np.asarray(rows)
        ordinary = np.array([int(r) not in self.other for r in rows])
        out = torch.zeros((len(rows), 512), dtype=torch.float32, device='cuda')
        if ordinary.any():
            index = torch.as_tensor(rows[ordinary], device='cuda')
            value = tail(self.standard[index].float()).mean((2, 3))
            out = out.index_copy(0, torch.as_tensor(np.flatnonzero(ordinary), device='cuda'), value)
        for pos in np.flatnonzero(~ordinary):
            value = tail(self.other[int(rows[pos])][None].float()).mean((2, 3))
            out = out.index_copy(0, torch.as_tensor([pos], device='cuda'), value)
        return out


def tensor(value):
    return torch.as_tensor(value, dtype=torch.float32, device='cuda')


def run_training():
    bank = PrefixBank()
    net = network()
    tail = net.layer4
    initial = copy.deepcopy(tail.state_dict())
    frame = pd.read_csv(OUT/'manifest.csv', dtype={'mmsi': str})
    labels = frame.class_id.to_numpy()
    for task in TASKS:
        for seed in SEEDS:
            folder = OUT/task/str(seed)
            split = dict(np.load(folder/'split.npz'))
            with np.load(folder/'features.npz') as z:
                rows = z['rows']
                base = z['r50']
                original_extra = z['r18']
            loc = pd.Index(rows)
            base_gpu = tensor(base)
            with np.load(folder/'geometry.npz') as z:
                geo = {k:tensor(z['r18_'+k]) for k in ('center', 'scale', 'matrix', 'bias')}
            with np.load(folder/'frozen/S/heads.npz') as z:
                coef, bias = z['dual_coef'], z['dual_bias']
            for arm in ARMS:
                dest = folder/'updated'/arm
                dest.mkdir(parents=True, exist_ok=True)
                tail.load_state_dict(initial)
                tail.eval()
                head = torch.nn.Linear(192, 8).cuda()
                with torch.no_grad():
                    head.weight.copy_(tensor(coef.T)); head.bias.copy_(tensor(bias))
                train = split['source'] if arm == 'S' else np.r_[split['source'], split['source_add' if arm == 'S_U' else 'target_add']]
                optimizer = torch.optim.AdamW([dict(params=tail.parameters(), lr=1e-5),
                    dict(params=head.parameters(), lr=1e-3)], weight_decay=1e-4)
                rng = np.random.default_rng(seed)
                by_class = [train[labels[train] == c] for c in range(8)]
                history = []
                start = time.perf_counter()
                for step in range(STEPS):
                    sample = np.concatenate([rng.choice(v, 8, replace=len(v) < 8) for v in by_class])
                    extra = bank.apply(tail, sample)
                    low = ((extra-geo['center'])/geo['scale'])@geo['matrix']+geo['bias']
                    x = torch.cat([base_gpu[loc.get_indexer(sample)], low], dim=1)/np.sqrt(2.)
                    loss = F.cross_entropy(head(x), torch.as_tensor(labels[sample], device='cuda'))
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite tail loss')
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    gradient = torch.nn.utils.clip_grad_norm_(list(tail.parameters())+list(head.parameters()), 5.)
                    optimizer.step()
                    history.append(dict(step=step+1, loss=float(loss.detach()), gradient=float(gradient)))
                    if (step+1) % 32 == 0:
                        print('TRAIN', task, seed, arm, step+1, round(float(loss.detach()), 5), flush=True)
                all_extra = np.empty((len(rows), 512), np.float32)
                with torch.inference_mode():
                    for lo in range(0, len(rows), 128):
                        ids = rows[lo:lo+128]
                        all_extra[lo:lo+len(ids)] = bank.apply(tail, ids).cpu().numpy()
                low = ((all_extra-geo['center'].cpu().numpy())/geo['scale'].cpu().numpy())@geo['matrix'].cpu().numpy()+geo['bias'].cpu().numpy()
                feat = dict(r50=base, r18=low)
                with threadpool_limits(limits=2):
                    pred, heads = predictions(feat, loc.get_indexer(train), labels[train],
                        loc.get_indexer(split['support']), labels[split['support']], len(split['source']))
                query_pos = loc.get_indexer(split['query'])
                np.savez_compressed(dest/'predictions.npz', rows=split['query'],
                                    **{k:v[query_pos] for k,v in pred.items()})
                np.savez(dest/'heads.npz', **{m+'_'+k:v for m,h in heads.items() for k,v in h.items()})
                np.save(dest/'extra.npy', all_extra)
                torch.save({k:v.cpu() for k,v in tail.state_dict().items()}, dest/'layer4.pt')
                delta = sum(float((tail.state_dict()[k]-initial[k]).square().sum()) for k in initial if initial[k].is_floating_point())
                write(dest/'TRAINING.json', dict(steps=STEPS, elapsed_seconds=time.perf_counter()-start,
                    tail_parameter_change_l2=float(np.sqrt(delta)), history=history,
                    train_rows=train.tolist(), initialization='Same e206 layer4 and S-ridge head for every arm',
                    prediction_head='Balanced ridge refit, not training softmax head'))
                print('DONE', task, seed, arm, round(time.perf_counter()-start, 1), flush=True)
                del optimizer
            write(folder/'UPDATED_COMPLETE.json', dict(arms=ARMS, steps=STEPS))
    write(OUT/'PREDICTIONS_FROZEN.json', dict(tasks=TASKS, seeds=SEEDS, stages=['frozen', 'updated'],
        query_metrics_not_used_for_selection=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['cache', 'train'])
    args = parser.parse_args()
    {'cache': prepare_cache, 'train': run_training}[args.stage]()
