"""Cache actual CCREL/PCEA readouts, fixed checkpoint and source references."""
from pathlib import Path
import importlib.util
import json
import sys
import time

ROOT = Path(__file__).resolve().parent
PREV = ROOT.parent/'pcea_ports24_startpoint_20261117'
spec = importlib.util.spec_from_file_location('readout_probe_model', PREV/'model.py')
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
torch, np = m.torch, m.np


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def summarize(h, mask, region_mean=False):
    if region_mean:
        mean = (h*mask[..., None]).sum((2,3))/mask.sum((2,3))[..., None].clamp_min(1e-8)
        h = mean[:, :, None, None].expand_as(h)
    cells = []
    for row in (slice(0,5), slice(5,10)):
        for col in (slice(0,5), slice(5,10)):
            w = mask[:, :, row, col]
            cells.append((h[:, :, row, col]*w[..., None]).sum((2,3))/w.sum((2,3))[..., None].clamp_min(1e-8))
    return torch.stack(cells, 2)


@torch.no_grad()
def main():
    if (ROOT/'cache_complete.json').exists():
        print('Existing complete cache retained.'); return
    inputs = m.Inputs()
    net = m.Model(17, 'cross_mean')
    checkpoint = PREV/'fits/cross_mean_s17_sp0.1/last.pt'
    net.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    net.requires_grad_(False).eval()
    net.accelerate()
    net.exchange.disabled = True
    reference = net.reference(inputs.data(inputs.groups['references'], 'small'))
    current = {}
    handles = []
    for p in range(2):
        def capture_h(module, args, p=p):
            current[p] = args[0].detach()
        handles.append(net.exchange.output[p].register_forward_pre_hook(capture_h))

    def capture(module, args, out):
        trace = module.trace
        full = [current[p] for p in range(2)]
        module.no_detail = True
        module.forward(*args)
        summary = [current[p] for p in range(2)]
        module.no_detail = False
        module.trace = trace
        mask = args[1]*trace['active'][:, None, :, None, None]
        b = trace['normalized']*trace['active'][:, None, :, None, None]
        current['batch'] = dict(
            base=b.flatten(1).cpu().numpy(),
            summary=torch.stack([summarize(summary[p],mask[:,p]) for p in range(2)],1).flatten(1).cpu().numpy(),
            detail_mean=torch.stack([summarize(full[p],mask[:,p],True) for p in range(2)],1).flatten(1).cpu().numpy(),
            detail_grid=torch.stack([summarize(full[p],mask[:,p]) for p in range(2)],1).flatten(1).cpu().numpy(),
            active=trace['active'].any(1).cpu().numpy())
    handles.append(net.exchange.register_forward_hook(capture))
    train = np.asarray(json.loads((PREV/'training_ids.json').read_text()), np.int64)
    groups = dict(train=train, validation=inputs.groups['validation'])
    metadata = {}
    tick = time.perf_counter()
    for split, ids in groups.items():
        chunks = []
        for first in range(0,len(ids),12):
            part = ids[first:first+12]
            z,_ = net(inputs.data(part,'small' if split=='train' else 'validation'),
                      [inputs.rows[i] for i in part],reference,inputs.reference_rows)
            chunks.append(dict(current['batch'], logits=z.cpu().numpy()))
            if first % 600 == 0:
                write(ROOT/'progress.json',dict(phase='extract',split=split,completed=first+len(part),total=len(ids)))
                print('EXTRACT',split,first+len(part),len(ids),round(time.perf_counter()-tick,1),flush=True)
        arrays = {k:np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
        assert all(np.isfinite(v).all() for v in arrays.values())
        if split == 'validation':
            with np.load(PREV/'fits/cross_mean_s17_sp0.1/lastvalidation_off.npz') as old:
                assert np.array_equal(ids,old['ids'])
                error = float(abs(arrays['logits']-old['score']).max())
                assert error < 1e-4, error
        np.savez(ROOT/(split+'_local.npz'),ids=ids,label=inputs.y[ids],weight=inputs.weights[ids],**arrays)
        metadata[split] = [dict(index=int(i),id=inputs.rows[i]['id'],label=int(inputs.y[i]),
            port=inputs.rows[i]['port'],vessel=inputs.rows[i]['mmsi'],
            passage='_'.join(inputs.rows[i]['pass_group'].split('_')[:2])) for i in ids]
    metadata['references'] = [dict(index=int(i),vessel=inputs.rows[i]['mmsi'],
        passage='_'.join(inputs.rows[i]['pass_group'].split('_')[:2])) for i in inputs.groups['references']]
    write(ROOT/'metadata_local.json',metadata)
    write(ROOT/'cache_complete.json',dict(checkpoint=str(checkpoint),
        train=len(train),validation=len(groups['validation']),
        baseline='same final checkpoint, PCEA writeback disabled',
        base_dimensions=1728,extra_dimensions=3456,validation_replay_error=error,
        elapsed_seconds=time.perf_counter()-tick,test_access=False))
    for h in handles: h.remove()
    print('CACHE COMPLETE',round(time.perf_counter()-tick,1),flush=True)


if __name__ == '__main__':
    main()
