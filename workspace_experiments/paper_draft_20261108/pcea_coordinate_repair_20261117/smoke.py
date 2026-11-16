"""Check readout equivalence, source-only coordinates and trainable path gradients."""
import gc
import model

torch, np, full = model.torch, model.np, model.full


def main():
    inputs = full.Inputs()
    for split in ('small', 'validation'):
        inputs.prepare_geometry(split)
    records = {}
    ids, val = inputs.groups['small'][:12], inputs.groups['validation'][:12]
    for arm in model.ARMS:
        net = model.Model(17, arm)
        model.calibrate(net, inputs, 17)
        before = full.predict(net, inputs, val, 'validation')
        net.exchange.disabled = True
        original = full.predict(net, inputs, val, 'validation')
        net.exchange.disabled = False
        assert abs(before-original).max() < 1e-5
        # The original pool-after-projection and current projection-after-pool are linear equivalents.
        c = {}
        def check(module, args, result):
            legacy = model.old.CCREL(net.net.guide.value, False).to('cuda')
            with torch.no_grad():
                expected, _ = legacy(*args)
            c['equivalence'] = float((expected-result[0]).abs().max())
        handle = net.ccrel.register_forward_hook(check)
        full.predict(net, inputs, val[:1], 'validation')
        handle.remove()
        assert c['equivalence'] < 2e-5, c
        one = full.predict(net, inputs, val[:1], 'validation')
        assert abs(one-before[:1]).max() < 1e-4
        params = {n: p for n, p in net.named_parameters() if p.requires_grad}
        assert all(n.startswith(('exchange.', 'ccrel.')) for n in params)
        assert not any(n.startswith('exchange.reader.') for n in net.state_dict())
        opt = torch.optim.AdamW(params.values(), lr=1e-3, weight_decay=.01)
        steps = []
        for step in range(2):
            net.train(); opt.zero_grad(set_to_none=True)
            loss, _ = full.loss_for(net, inputs, ids, 1)
            loss.backward()
            grads = {n: float(p.grad.norm()) for n, p in params.items() if p.grad is not None}
            assert np.isfinite(list(grads.values())).all()
            assert grads['exchange.output.0.weight'] > 0
            if step:
                for name in ('ccrel.value.0.weight', 'exchange.project.0.weight', 'exchange.gate.0.weight'):
                    assert grads[name] > 0, (arm, name, grads[name])
            torch.nn.utils.clip_grad_norm_(list(params.values()), 5.)
            opt.step()
            steps.append(dict(loss=float(loss.detach()), gradients=grads))
        actual = full.predict(net, inputs, val, 'validation')
        single = full.predict(net, inputs, val[:1], 'validation')
        assert abs(single-actual[:1]).max() < 1e-4
        net.exchange.disabled = True
        withdrawn = full.predict(net, inputs, val, 'validation')
        assert abs(withdrawn-original).max() < 1e-5
        with torch.no_grad():
            evidence = net.ccrel.project_raw(net.source_raw)
            net.exchange.source_raw = net.source_raw
            net.exchange.source_weight = net.source_weight
            object.__setattr__(net.exchange, 'reader', net.ccrel)
            t, _ = net.exchange.standardized(evidence)
            w = net.source_weight[:, None, :, None, None]
            weighted_mean = (t*w).sum(0)/w.sum(0)
            assert weighted_mean.abs().max() < 1e-4
        records[arm] = dict(initial_error=float(abs(before-original).max()),
            pool_projection_equivalence=c['equivalence'], steps=steps,
            trained_batch_single_error=float(abs(single-actual[:1]).max()),
            current_coordinate_source_mean_max=float(weighted_mean.abs().max()),
            withdrawal_error=float(abs(withdrawn-original).max()))
        print('SMOKE', arm, records[arm]['current_coordinate_source_mean_max'], flush=True)
        del net, opt, params
        gc.collect(); torch.cuda.empty_cache()
    full.write(model.ROOT/'smoke.json', records)


if __name__ == '__main__':
    main()
