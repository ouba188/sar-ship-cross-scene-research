"""Forward-only local-detail ablation and actual writeback amplitudes."""
from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('ports24_probe_model', ROOT/'model.py')
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
np, torch = m.np, m.torch


def main():
    import json
    inputs = m.Inputs()
    ids = inputs.groups['validation']
    folder = ROOT/'fits/cross_mean_s17_sp0.1'
    net = m.Model(17, 'cross_mean')
    net.load_state_dict(torch.load(folder/'last.pt', map_location='cpu', weights_only=True), strict=True)
    net.accelerate()
    records = []

    def capture(module, args, out):
        t = module.trace
        records.append({k:t[k].cpu().numpy() for k in (
            'tokens', 'normalized', 'active', 'detail_rms', 'context_spatial_rms',
            'token_spatial_rms', 'input_rms', 'update_rms')})

    hook = net.exchange.register_forward_hook(capture)
    on = m.full.predict(net, inputs, ids, 'validation')
    hook.remove()
    with np.load(folder/'lastvalidation.npz') as z:
        assert np.array_equal(ids, z['ids'])
        replay_error = float(abs(on-z['score']).max())
        assert replay_error < 1e-4
    trace = {k:np.concatenate([r[k] for r in records]) for k in records[0]}
    assert len(trace['tokens']) == len(ids)
    net.exchange.no_detail = True
    nodetail = m.full.predict(net, inputs, ids, 'validation')
    out = ROOT/'decomposition'
    np.savez_compressed(out/'probe_local.npz', ids=ids, label=inputs.y[ids], on=on,
                        no_detail=nodetail, **trace)
    active = trace['active'].any(1)
    relative = trace['update_rms'][active]/np.maximum(trace['input_rms'][active], 1e-12)
    def stats(v):
        return dict(median=float(np.median(v)), p95=float(np.quantile(v,.95)), max=float(np.max(v)))
    delta = on[active]-nodetail[active]
    delta -= delta.mean(1, keepdims=True)
    result = dict(checkpoint='cross_mean_s17_sp0.1/last.pt', training_steps=0,
        replay_error=replay_error, active_samples=int(active.sum()),
        update_to_input_rms=stats(relative),
        within_region_token_spatial_rms=stats(trace['token_spatial_rms'][active]),
        context_spatial_rms=stats(trace['context_spatial_rms'][active]),
        local_detail_rms=stats(trace['detail_rms'][active]),
        no_detail_logit_centered_rms=float(np.sqrt(np.mean(delta**2))),
        no_detail_max_logit_difference=float(abs(on-nodetail).max()),
        no_detail_prediction_changes=int((on.argmax(1)!=nodetail.argmax(1)).sum()))
    with np.load(folder/'lastvalidation_off.npz') as z:
        off = z['score']
    cases = []
    for j, idx in enumerate(ids):
        if str(inputs.rows[idx]['id']) not in ('21433','35906','31926','23235','35666'):
            continue
        y = int(inputs.y[idx])
        def margin(v):
            wrong = v.copy();wrong[y]=-np.inf
            return float(v[y]-wrong.max())
        cases.append(dict(id=inputs.rows[idx]['id'], on_margin=margin(on[j]),
            off_margin=margin(off[j]), no_detail_margin=margin(nodetail[j]),
            update_to_input_rms=(trace['update_rms'][j]/trace['input_rms'][j]).tolist()))
    result['cases'] = cases
    (out/'probe_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
