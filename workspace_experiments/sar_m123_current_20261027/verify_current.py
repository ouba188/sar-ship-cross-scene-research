"""Verify exported code from raw chips without importing any research runner."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sar_m123.current.data import read_manifest, prepare_record, batch, pair_mask
from sar_m123.current.interventions import select_records
from sar_m123.current.model import load_model, predict_records, setup, reference_bank, forward_batch, training_policy, LIVE
from sar_m123.current.training import adaptation_loss, fitting_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--device', default='cuda')
    p.add_argument('--selection', choices=('none', 'smoke', 'all'), default='smoke')
    args = p.parse_args()
    root, device = args.root, args.device
    setup(device)
    refs = read_manifest(root/'manifests/references.json')
    demo = read_manifest(root/'manifests/demo.json')
    expected = json.loads((root/'verification/expected_demo.json').read_text(encoding='utf-8'))
    assert [str(r['id']) for r in demo] == expected['ids']
    receipt = dict(historical_imports=False, demo_count=len(demo), logits_max_errors={})
    for name, target in expected['scores'].items():
        net = load_model(root/f'weights/{name}.pt', device)
        score = predict_records(net, demo, refs, device).numpy()
        error = float(np.abs(score-np.asarray(target)).max())
        assert error < 1e-4, (name, error)
        receipt['logits_max_errors'][name] = error
        print('REPLAY', name, error, flush=True)
        del net
    net = load_model(root/'weights/teacher.pt', device)
    if args.selection != 'none':
        selection_path = 'source_adapt' if args.selection == 'all' else 'selection_smoke'
        rows = read_manifest(root/f'manifests/{selection_path}.json')
        result = select_records(net, rows, refs, device,
            lambda n, count, r: print(f'SELECTION {n}/{count} {r["id"]} {r["useful"]}', flush=True))
        old = {r['id']: r for r in json.loads((root/'manifests/selection_archived.json').read_text(encoding='utf-8'))}
        for r in result:
            assert r['useful'] == old[r['id']]['useful'], r['id']
            assert r['choice']['chosen_index'] == old[r['id']]['choice']['chosen_index'], r['id']
            for key in ('original', 'altered'):
                assert np.max(abs(np.array(r[key])-old[r['id']][key])) < 1e-4, (r['id'], key)
        receipt['selection'] = dict(n=len(result), useful=sum(r['useful'] for r in result), all_masks_and_choices_match=True)
        (root/'verification/selection_recomputed.json').write_text(json.dumps(result, indent=2), encoding='utf-8')

    rows = read_manifest(root/'manifests/selection_smoke.json')[:2]
    values = batch([prepare_record(r) for r in rows], device)
    bank = reference_bank(net, refs, device)
    legal = pair_mask(rows, refs, device)
    y = torch.tensor([r['label'] for r in rows], device=device)
    with torch.no_grad():
        teacher = forward_batch(net, values, bank, legal).clone()
    for stage in ('adapt', 'fit'):
        training_policy(net, stage)
        net.zero_grad(set_to_none=True)
        score, detail = forward_batch(net, values, bank, legal, True)
        w = torch.ones(len(rows), device=device)
        if stage == 'adapt':
            loss, _ = adaptation_loss(score, y, teacher, w, torch.ones_like(w), .3683066666126251)
        else:
            loss, _ = fitting_loss(score, detail, y, w, torch.tensor([r['label'] for r in refs], device=device))
        loss.backward()
        active = {n: float(p.grad.norm()) for n, p in net.named_parameters() if p.grad is not None}
        assert all(n in active and active[n] > 0 for n in LIVE)
        if stage == 'adapt':
            assert set(active) == set(LIVE)
            assert sum(p.numel() for p in net.parameters() if p.requires_grad) == 1114112
        else:
            for prefix in ('internal_m1.', 'internal_m2.', 'm3.', 'head.'):
                assert any(n.startswith(prefix) and v > 0 for n, v in active.items()), prefix
        receipt[stage] = dict(trainable=sum(p.numel() for p in net.parameters() if p.requires_grad),
                              finite_loss=float(loss.detach()), gradient_parameter_count=len(active))
        assert all(not m.training for m in net.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm))
        print('GRADIENT', stage, receipt[stage], flush=True)
    (root/'verification/receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print('VERIFICATION COMPLETE', flush=True)


if __name__ == '__main__':
    main()
