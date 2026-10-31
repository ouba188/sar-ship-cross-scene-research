"""Real-chip engineering verification, not a classification benchmark."""
import argparse
import gc
import json
from pathlib import Path
import sys

import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sar_m123.current.data import FREQUENCY_KEYS, batch, pair_mask, prepare_record, read_manifest
from sar_m123.current.interventions import select_records
from sar_m123.current.model import forward_batch, load_model, reference_bank, save_model, setup
from sar_m123.current.training import train


def compare(actual, expected, tolerance=1e-4):
    error = float((actual - expected).abs().max())
    if not torch.isfinite(actual).all() or error > tolerance:
        raise AssertionError((error, tolerance))
    return error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    setup(args.device)
    root = args.root.resolve()
    out = root/'verification'
    out.mkdir(exist_ok=True)
    refs = read_manifest(root/'manifests/references.json')
    case_ids = {'12457', '30167', '31141', '34278'}
    rows = [r for r in read_manifest(root/'manifests/development.json') if str(r['id']) in case_ids]
    assert len(rows) == len(case_ids), 'Expected the four predeclared engineering cases.'
    (root/'manifests/engineering_cases.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    values = batch([prepare_record(r) for r in rows], args.device)
    legal = pair_mask(rows, refs, args.device)
    receipt = dict(scope='Engineering integration only; no OA/BA claim.',
                   queries=[r['id'] for r in rows], reference_count=len(refs))
    old = load_model(args.previous, args.device)
    with torch.no_grad():
        old_bank = reference_bank(old, refs, args.device)
        old_score = forward_batch(old, values, old_bank, legal)
    del old, old_bank
    gc.collect()
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()

    net = load_model(root/'weights/current.pt', args.device)
    with torch.no_grad():
        bank = reference_bank(net, refs, args.device)
        score, detail = forward_batch(net, values, bank, legal, True)
        receipt['layer1_reference_cache'] = list(bank['prefix'].shape)
        receipt['query_m2_update_l1'] = [float(d['update'].abs().sum())
                                       for d in detail['operators']['layer2_block2']]
        receipt['reference_m2_update_l1'] = [float(d['update'].abs().sum())
                                           for d in detail['operators']['reference_layer2_block2']]
        assert all(v > 0 for v in receipt['query_m2_update_l1'] + receipt['reference_m2_update_l1'])
        for writer in net.layer2_m2:
            writer.enabled = False
        disabled = forward_batch(net, values, bank, legal)
        receipt['disabled_vs_previous_max_error'] = compare(disabled, old_score)
        for writer in net.layer2_m2:
            writer.enabled = True
        receipt['enabled_vs_disabled_max_difference'] = float((score-disabled).abs().max())
        assert receipt['enabled_vs_disabled_max_difference'] > 1e-6
        modified = dict(values)
        modified_bank = dict(bank)
        for key in FREQUENCY_KEYS:
            modified[key] = values[key]*13 + 17
            modified_bank[key] = bank[key]*7 + 3
        receipt['frequency_input_max_error'] = compare(
            forward_batch(net, modified, modified_bank, legal), score, 0.)
        direct = net(values['images'], values['masks'], values['valid'], values['available'],
                     reference=bank, pair_valid=legal,
                     **{k: v for k, v in values.items()
                        if k not in ('images', 'masks', 'valid', 'available')})
        receipt['native_vs_cached_max_error'] = compare(direct, score)
        save_model(net, out/'roundtrip.pt', purpose='engineering_roundtrip')
    del bank, detail, net, direct, modified_bank
    gc.collect()
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()
    net = load_model(out/'roundtrip.pt', args.device)
    with torch.no_grad():
        bank = reference_bank(net, refs, args.device)
        receipt['reload_max_error'] = compare(forward_batch(net, values, bank, legal), score, 0.)
    receipt['cases'] = [dict(id=r['id'], label=r['label'],
        previous_prediction=int(a.argmax()), integrated_prediction=int(b.argmax()),
        max_logit_change=float((a-b).abs().max())) for r, a, b in zip(rows, old_score, score)]
    panels = []
    for row in rows + refs[:2]:
        _, views = prepare_record(row, True)
        panel = Image.new('RGB', (520, 195), 'white')
        draw = ImageDraw.Draw(panel)
        draw.text((5, 3), f'ID {row["id"]} | label {row.get("label")} | native160, no resize', fill='black')
        for p, view in enumerate(views):
            draw.text((5+p*260, 21), ('VV', 'VH')[p], fill='black')
            if view is not None:
                panel.paste(Image.fromarray(view['pixels']).convert('RGB'), (5+p*260, 34))
        panels.append(panel)
    gallery = Image.new('RGB', (520, 195*len(panels)), 'white')
    for i, panel in enumerate(panels):
        gallery.paste(panel, (0, i*195))
    gallery.save(out/'native_cases.png')
    del bank, values, modified, score, old_score
    gc.collect()

    # These single-step runs only check the supported fit/select/adapt interfaces.
    fit_rows = read_manifest(root/'manifests/source_fit.json')[:2]
    receipt['fit_smoke'] = train(net, fit_rows, refs, out/'fit_smoke', args.device,
                                 stage='fit', epochs=1, batch_size=2, max_steps=1)
    source_rows = read_manifest(root/'manifests/selection_smoke.json')[:2]
    net.eval().requires_grad_(False)
    selection = select_records(net, source_rows, refs, args.device)
    (out/'selection_smoke.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    receipt['selection_smoke'] = dict(count=len(selection), useful=sum(r['useful'] for r in selection))
    receipt['adapt_smoke'] = train(net, source_rows, refs, out/'adapt_smoke', args.device,
        stage='adapt', selection=selection, epochs=1, batch_size=2, max_steps=1)
    receipt['status'] = 'passed'
    (out/'ENGINEERING_RECEIPT.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
