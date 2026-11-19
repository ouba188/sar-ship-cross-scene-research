"""Verify the merged paths, not classification performance. No optimizer steps."""
import json
import matched_run as runner

base, torch, np = runner.base, runner.torch, runner.np
OUT = runner.output_dir('unified')


def diff(a,b):
    return float((a-b).detach().abs().max())


def main():
    bank = torch.load(OUT/'verification_bank_local.pt',map_location='cpu',weights_only=True)
    assert bank['model_variant'] == 'unified'
    inputs, _, _ = base.runtime('unified')
    # Same two source records across all switches, independent of predicted outcome.
    ids = bank['ids'][[0,12]].numpy()
    rows = [inputs.rows[i] for i in ids]
    data = inputs.data(ids,'small')
    rd = inputs.data(inputs.groups['references'],'small')
    assert data['read40'].shape[-2:] == (40,40)
    assert data['read20'].shape[-2:] == (20,20)
    assert data['fine_write'].shape[-2:] == (10,10)
    net = base.model('unified')
    batchnorm = {name:value.detach().clone() for name,value in net.named_buffers()
                 if name.endswith(('running_mean','running_var','num_batches_tracked'))}
    scores, modes = {}, {}
    for train in (False,True):
        net.train(train)
        with torch.no_grad():
            ref = net.reference(rd)
            z,_ = net(data,rows,ref,inputs.reference_rows)
            modes[str(train)] = z.detach()
    replay = diff(modes['False'],modes['True'])
    assert replay < 1e-4, replay
    net.eval()
    with torch.no_grad():
        for selector in ('soft','coarse','fine'):
            net.selection_mode = selector
            ref = net.reference(rd)
            for pcea in (False,True):
                net.exchange.disabled = not pcea
                for m3 in (False,True):
                    z,_ = net(data,rows,ref,inputs.reference_rows,m3_on=m3)
                    scores[f'{selector}/pcea{int(pcea)}/m3{int(m3)}'] = z.detach()
    isolation = {}
    for pcea in (0,1):
        expected = scores[f'soft/pcea{pcea}/m30']
        isolation[str(pcea)] = max(diff(expected,scores[f'{s}/pcea{pcea}/m30']) for s in ('coarse','fine'))
        assert isolation[str(pcea)] < 1e-5, isolation
    m3_effect = diff(scores['soft/pcea1/m31'],scores['soft/pcea1/m30'])
    pcea_effect = diff(scores['soft/pcea1/m31'],scores['soft/pcea0/m31'])
    m3_without_pcea = diff(scores['soft/pcea0/m31'],scores['soft/pcea0/m30'])
    pcea_without_m3 = diff(scores['soft/pcea1/m30'],scores['soft/pcea0/m30'])
    granularity_effect = diff(scores['coarse/pcea1/m31'],scores['fine/pcea1/m31'])
    assert m3_effect > 1e-7 and pcea_effect > 1e-7 and granularity_effect > 1e-7
    net.selection_mode = 'soft'
    net.exchange.disabled = False
    net.train()
    net.zero_grad(set_to_none=True)
    reference = net.reference(rd)
    reference['desc'].retain_grad()
    z,aux = net(data,rows,reference,inputs.reference_rows)
    labels = torch.as_tensor(inputs.y[ids],device='cuda')
    loss = torch.nn.functional.cross_entropy(z,labels)
    loss.backward()
    reference_gradient = float(reference['desc'].grad.norm())
    assert reference_gradient > 0
    parameters = dict(net.named_parameters())
    required = ['net.matcher.position_strength_raw','net.matcher.key.0.weight','net.matcher.value.0.weight']
    required.extend(name for name in parameters if name.startswith('selector.'))
    gradients = {}
    for name in required:
        assert name in parameters, name
        grad = parameters[name].grad
        assert grad is not None and bool(torch.isfinite(grad).all()), name
        gradients[name] = float(grad.norm())
    assert gradients and all(np.isfinite(value) and value > 0 for value in gradients.values()),gradients
    groups = base.optimizer_groups(net,'unified')
    optimizer = torch.optim.AdamW(groups,weight_decay=.01)
    assert len(optimizer.param_groups) == 5
    for name,value in net.named_buffers():
        if name in batchnorm:
            assert torch.equal(value,batchnorm[name]),name
    result = dict(passed=True,optimizer_steps=0,queries=len(ids),references=len(inputs.groups['references']),
        train_eval_soft_replay_error=replay,selector_effect_when_m3_off=isolation,
        m3_logit_effect=m3_effect,pcea_logit_effect=pcea_effect,granularity_logit_effect=granularity_effect,
        m3_effect_without_pcea=m3_without_pcea,pcea_effect_without_m3=pcea_without_m3,
        live_reference_gradient=reference_gradient,specific_classification_gradients=gradients,
        optimizer_groups=[g['name'] for g in groups],bn_statistics_unchanged=True,
        peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3)
    (OUT/'integration.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (OUT/'switches_local.json').write_text(json.dumps(dict(ids=ids.tolist(),labels=inputs.y[ids].tolist(),
        source_names=[r['id'] for r in rows],scores={k:v.cpu().tolist() for k,v in scores.items()}),indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
