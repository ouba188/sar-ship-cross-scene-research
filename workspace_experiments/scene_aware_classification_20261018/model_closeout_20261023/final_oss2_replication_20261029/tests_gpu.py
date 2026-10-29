"""Deferred real-input checks. No tensor library imports until check_gpu is called."""
from protocol import HERE, INIT, CLASS_INDICES, CLASSES, write


def check_gpu(run, rows, cache, device):
    torch, np, api = run.torch, run.np, run.api
    data, model_api, training = run.data, run.model_api, run.training
    from sar_m123.pool_replacement import PoolReplacementResNet50
    from torch.nn import functional as F
    refs, queries = rows['references'], run.sample_rows(rows)
    bank = cache.bank(refs)
    assert any(r['vv'] is None for r in queries) and any(r['vh'] is None for r in queries)
    assert {r['label'] for r in queries} == set(range(4))
    source = torch.load(INIT, map_location='cpu', weights_only=True)['state']
    checks = dict(status='running', class_indices=list(CLASS_INDICES), classes=list(CLASSES),
                  sample_ids=[r['id'] for r in queries], arms={})
    write(HERE/'gpu_checks.json', checks)
    legal = data.pair_mask([refs[0], dict(refs[0], id='different-observation')], refs, device)
    assert not legal[:, 0].any()
    # Validate native geometry/availability on representative actual chips, including a large chip.
    native_items = []
    for record in queries:
        item, views = data.prepare_record(record, return_views=True)
        assert item['images'].shape == (2, 1, 160, 160)
        assert item['images'].dtype == np.float32
        assert item['available'].tolist() == [float(record[p] is not None) for p in ('vv', 'vh')]
        for p, view in enumerate(views):
            if record[('vv', 'vh')[p]] is None:
                assert view is None and not item['valid'][p].any() and not item['images'][p].any()
            else:
                place = view['placement']
                assert place['input_scale'] == 1 and view['valid_target']
                assert place['native_target_pixels'] == place['retained_target_pixels']
        native_items.append(item)
    native_reference = None
    for arm in ('full', 'plain'):
        net = run.fresh(arm, device)
        assert net.arm == arm and net.head.out_features == 4
        assert net.head.in_features == (7170 if arm == 'full' else 4098)
        state = net.state_dict()
        for name, value in state.items():
            expected = source[name]
            if name.startswith('head.') or (name.startswith('internal_m1.') and '.reader.' in name):
                expected = expected[list(CLASS_INDICES)]
            if name == 'head.weight' and arm == 'plain':
                expected = torch.cat((expected[:, :2048], expected[:, 3584:5632], expected[:, -2:]), 1)
            assert torch.equal(value.cpu(), expected), ('fresh_init_mapping', arm, name)
        if arm == 'full':
            assert all(layer.reader.out_channels == 4 for layer in net.internal_m1)
        else:
            assert not net.internal_m1 and not net.internal_m2 and not net.m3
        with torch.no_grad():
            input_errors = {}
            original_batch_errors = []
            for record in queries:
                start = cache.lookup[record['id']]//8*8
                group = cache.records[start:start+8]
                original_values = data.batch([data.prepare_record(r) for r in group], device)
                original_prefix = net.prefix(original_values['images'], stop=2)
                err = float((original_prefix-cache.batch(group)['prefix']).abs().max())
                assert err < 1e-5, ('original_cache_batch', arm, record['id'], err)
                original_batch_errors.append(err)
            for first in range(0, len(queries), 3):
                group = queries[first:first+3]
                native = data.batch(native_items[first:first+3], device)
                native['prefix'] = net.prefix(native.pop('images'), stop=2)
                cached = cache.batch(group)
                for key in native:
                    err = float((native[key]-cached[key]).abs().max())
                    input_errors[key] = max(input_errors.get(key, 0.), err)
                    if key != 'prefix':
                        assert err < 1e-7, (arm, key, err)
                    else:
                        # Different batch sizes can change FP32 convolution
                        # reduction order; check cache identity and final logits.
                        assert torch.isfinite(native[key]).all() and torch.isfinite(cached[key]).all()
            native_reference = model_api.reference_bank(net, refs, device)
            reference_error = float((native_reference['prefix']-bank['prefix']).abs().max())
            assert torch.isfinite(native_reference['prefix']).all()
        native_error = run.native_error(net, cache, queries, refs, bank)
        model_api.training_policy(net, 'fit')
        fit_trainable = {n for n, p in net.named_parameters() if p.requires_grad}
        assert all(n in fit_trainable for n in model_api.LIVE)
        assert not any(m.training for m in net.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm))
        if arm == 'plain':
            assert fit_trainable == set(model_api.LIVE) | {'head.weight', 'head.bias'}
        else:
            assert not any('.reader2.' in n or '.reader3.' in n for n in fit_trainable)
        fit_count = sum(p.numel() for p in net.parameters() if p.requires_grad)
        q = queries[:4]
        score, detail = cache.score(net, q, refs, bank, True)
        assert score.shape == (len(q), 4)
        assert len(detail['aux_logits']) == (1 if arm == 'full' else 0)
        assert len(detail['relations']) == (2 if arm == 'full' else 0)
        labels = torch.tensor([r['label'] for r in q], device=device)
        weights = torch.ones(len(q), device=device)
        ref_labels = torch.tensor([r['label'] for r in refs], device=device)
        loss, terms = api.fitting_loss(score, detail, labels, weights, ref_labels)
        if arm == 'plain':
            assert terms['auxiliary'].item() == terms['relation'].item() == 0
            torch.testing.assert_close(loss, F.cross_entropy(score, labels))
        assert torch.isfinite(loss)
        loss.backward()
        grads = {n: float(p.grad.norm()) for n, p in net.named_parameters() if p.grad is not None}
        assert all(grads.get(n, 0) > 0 for n in model_api.LIVE)
        assert all(p.grad is None for n, p in net.named_parameters() if n not in fit_trainable)
        if arm == 'full':
            for prefix in ('internal_m1.', 'internal_m2.', 'm3.', 'head.'):
                assert any(n.startswith(prefix) and g > 0 for n, g in grads.items()), prefix
        else:
            ordinary = PoolReplacementResNet50(None, mode='native', classes=4).to(device).eval()
            ordinary.backbone.load_state_dict(net.backbone.state_dict())
            ordinary.head.load_state_dict(net.head.state_dict())
            value = cache.batch(q)
            with torch.no_grad():
                expected = ordinary.forward_from(*(value[k] for k in ('prefix', 'masks', 'valid', 'available')))
            torch.testing.assert_close(score.detach(), expected, atol=1e-5, rtol=0)
            with torch.no_grad():
                without_reference = net.forward_from(**value, reference=None, pair_valid=None)
            torch.testing.assert_close(score.detach(), without_reference, atol=1e-5, rtol=0)
            del ordinary
        net.zero_grad(set_to_none=True)
        model_api.training_policy(net, 'adapt')
        assert {n for n, p in net.named_parameters() if p.requires_grad} == set(model_api.LIVE)
        score = cache.score(net, q, refs, bank)
        teacher = score.detach().clone()
        useful = torch.tensor([arm == 'full']*len(q), device=device)
        adapt_loss, _ = training.adaptation_loss(score, labels, teacher, weights, useful,
                                               run.COEFFICIENT if arm == 'full' else 0.)
        adapt_loss.backward()
        assert all(p.grad is None for n, p in net.named_parameters() if n not in model_api.LIVE)
        assert all(net.get_parameter(n).grad.norm().item() > 0 for n in model_api.LIVE)
        saved = HERE/'checks'/f'{arm}_four_class.pt'
        api.save(net, saved, classes_names=list(CLASSES), class_indices=list(CLASS_INDICES), smoke_only=True)
        restored = api.load(saved, device)
        assert restored.arm == arm and restored.head.out_features == 4
        assert all(torch.equal(restored.state_dict()[k], v) for k, v in net.state_dict().items())
        with torch.no_grad():
            replay = cache.score(restored, q, refs, bank)
        save_error = float((replay-score.detach()).abs().max())
        assert save_error < 1e-5
        checks['arms'][arm] = dict(fit_trainable=fit_count, native_cache_error=native_error,
            input_errors=input_errors, reference_prefix_error=reference_error,
            original_batch_prefix_errors=original_batch_errors,
            save_replay_error=save_error, gradients=grads, head_rows_verified=True,
            reader_rows_verified=arm == 'full', exact_live_policy=True)
        del net, restored, score, detail, loss, adapt_loss, state, native_reference
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
    # Four-class margin denominator is 3; plain CE+KL has no margin value or gradient.
    labels = torch.tensor([0, 1], device=device)
    student = torch.tensor([[1., 0., -.4, .2], [.1, .5, .2, -.3]], device=device, requires_grad=True)
    teacher = torch.tensor([[2., .1, 0., -.1], [.5, .1, .2, -.2]], device=device)
    weights = torch.tensor([.5, 1.5], device=device)
    useful = torch.ones(2, dtype=torch.bool, device=device)
    floor = training.margin_floor(student, teacher, labels, weights, useful)
    expected = sum(weights[i]*sum(torch.relu((teacher[i, labels[i]]-teacher[i, c])-
                    (student[i, labels[i]]-student[i, c]))**2 for c in range(4))/3
                   for i in range(2))/weights.sum()
    torch.testing.assert_close(floor, expected)
    actual, terms = training.adaptation_loss(student, labels, teacher, weights, useful*False, 0.)
    ce = (weights*F.cross_entropy(student, labels, reduction='none')).sum()/weights.sum()
    kl = F.kl_div(student.log_softmax(1), teacher.log_softmax(1), reduction='none', log_target=True).sum(1)
    expected = ce+.1*(weights*teacher.argmax(1).eq(labels)*kl).sum()/weights.sum()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, student, retain_graph=True)[0],
                               torch.autograd.grad(expected, student, retain_graph=True)[0])
    assert terms['floor'].item() == 0
    logits = torch.ones(2, 32, device=device)
    empty_valid = torch.zeros(2, 32, dtype=torch.bool, device=device)
    empty_loss, empty_terms = api.fitting_loss(student, dict(aux_logits=[], relations=[(logits, empty_valid)]),
        labels, weights, torch.tensor([r['label'] for r in refs], device=device))
    torch.testing.assert_close(empty_loss, (F.cross_entropy(student, labels, reduction='none')*weights).mean())
    assert empty_terms['relation'].item() == empty_terms['auxiliary'].item() == 0
    checks.update(status='passed', empty_branches=True, four_class_margin_denominator=3,
                  plain_ce_kl_value_and_gradient=True, native160_and_polarization=True)
    write(HERE/'gpu_checks.json', checks)
    print('OSS2_GPU_CHECKS_PASS', flush=True)
