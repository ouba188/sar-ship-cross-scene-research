"""M3 operator checks and optional legacy parity validation (not inference code).

Run basic tests: python -m unittest discover -s tests -p test_m3.py
Golden check: python tests/test_m3.py --legacy-root PATH --assets-root PATH
The golden route reads query caches only as an independent regression fixture.
"""
import argparse
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from sar_m123 import frequency, tokens
from sar_m123.m3 import M3, AnchoredGate, PolarHead, sinkhorn


class OperatorTests(unittest.TestCase):
    def test_fractional_pixel_mass(self):
        impulse = np.zeros((160, 160))
        impulse[0, 2] = 1
        actual = tokens.cell_mass(impulse)
        self.assertEqual(actual[0, 0], .5)
        self.assertEqual(actual[0, 1], .5)
        self.assertEqual(actual.sum(), 1)

    def test_slots_partition_regions(self):
        g = np.array([73.2, 81.7, 27., 9., .37, .8])
        fp = np.ones((160, 160))
        np.testing.assert_allclose(tokens.slots(g, fp).sum(0), tokens.soft_regions(g).sum(0), atol=1e-12)

    def test_dc_and_native_placement(self):
        raw = np.ones((53, 71))
        self.assertLess(np.abs(frequency.response(raw, frequency.kernel(8., .37))).max(), 1e-11)
        p = dict(crop_x=0, crop_y=0, crop_width=71, crop_height=53, pad_left=45, pad_top=54)
        placed = frequency.place(raw, p)
        self.assertEqual(placed.sum(), raw.sum())
        self.assertEqual(placed[54, 45], 1)

    def test_ot_matches_and_rejects(self):
        sim = torch.tensor([[[1., -1.], [-1., 1.]], [[-1., -1.], [-1., -1.]]])
        t = sinkhorn(sim, torch.ones(2, 2), torch.ones(2, 2), steps=300)
        self.assertGreater(float(t[0].trace()), .9)
        self.assertLess(float(t[1].sum()), .02)

    def test_null_center_and_missing_gate(self):
        head = PolarHead(1, True, np.linspace(-1, 1, 15).astype(np.float32))
        x = torch.randn(3, 8, 15)
        x[..., :12] = head.null[:12]
        self.assertEqual(float(head(x).abs().max()), 0)
        gate = AnchoredGate(.6)
        with torch.no_grad():
            gate.net[-1].bias.fill_(100)
        available = torch.tensor([[1., 1.], [1., 0.], [0., 0.]])
        weights, alpha = gate(torch.randn(3, 8, 185), available)
        np.testing.assert_allclose(alpha.detach(), [.85, .6, .6], atol=1e-6)
        np.testing.assert_allclose(weights.detach(), [[.85, .15], [.6, 0], [0, 0]], atol=1e-6)


def legacy_parity(legacy_root, assets_root, device='cpu', limit=None):
    root = Path(legacy_root)
    local = root/'local_evidence_transport'
    redesign = root/'evidence_preserving_redesign'
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = M3(assets_root, device=device)
    original = np.load(local/'tokens.npz')
    j = np.load(local/'spatial_frequency_joint_pooling/features.npz')
    f = np.load(local/'frequency_conditioned_matching/features.npz')
    old_refs = np.load(local/'scores_layer1.npz')['refs']
    order_lookup = {int(v): i for i, v in enumerate(j['ids'])}
    records = []
    for seed in (17, 29, 43):
        cached = np.load(redesign/'polar_readout_diagnosis_20261025/results'/f'inputs_{seed}.npz')
        gate = np.load(redesign/'query_polar_gate_20261025/results'/f'features_{seed}.npz')
        final = np.load(redesign/'anchored_polar_adjustment_20261025/results'/f'frequency_{seed}.npz')
        ids = cached['up_index']
        if limit is not None:
            ids = ids[:limit]
        order = np.array([order_lookup[int(v)] for v in ids])
        mass = original['mass'][ids]
        x = tokens.unit((original['layer1'][ids]-model.projection['mean'])@model.projection['components'].T).astype(np.float32)
        x[mass <= 0] = 0
        features = dict(x=x, mass=mass, mean=j['mean'][order], joint=j['joint'][order],
                        e_image=f['e_image'][order], coverage=f['coverage'][order])
        errors = dict(local=0., delta_vv=0., delta_vh=0., gate_raw=0., final_score=0., weight=0.)
        changed = 0
        for start in range(0, len(ids), 24):
            sl = slice(start, min(start+24, len(ids)))
            batch = {k: v[sl] for k, v in features.items()}
            score, details = model.forward(batch, cached['base'][sl], seed)
            np.testing.assert_array_equal(details['reference_ids'], old_refs[ids[sl]])
            expected = dict(local=cached['local'][sl], delta_vv=gate['delta_vv'][sl],
                            delta_vh=gate['delta_vh'][sl], gate_raw=gate['raw'][sl],
                            final_score=final['score'][sl], weight=final['weight'][sl])
            actual = dict(details, final_score=score, weight=details['polar_weights'])
            for key in errors:
                errors[key] = max(errors[key], float(abs(actual[key]-expected[key]).max()))
            changed += int((score.argmax(1) != final['score'][sl].argmax(1)).sum())
        assert errors['local'] < 3e-5, errors
        assert errors['gate_raw'] < 5e-5, errors
        assert errors['final_score'] < 5e-5, errors
        assert changed == 0, changed
        record = dict(seed=seed, n=len(ids), changed_predictions=changed, max_abs_errors=errors,
                      all_retrieved_references_equal=True)
        records.append(record)
        print(json.dumps(record), flush=True)
    return records


def extraction_parity(legacy_root, assets_root, device='cpu'):
    """Real-chip extraction check, independent of cached query feature input."""
    import importlib.util
    import pandas as pd
    from PIL import Image
    from sar_m123.backbone import Backbone

    root = Path(legacy_root)
    local = root/'local_evidence_transport'
    source = root.parent/'eight_class_evidence_20261023/obb_dynamic_readout/common.py'
    spec = importlib.util.spec_from_file_location('legacy_geometry_test_only', source)
    geo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geo)
    table = pd.read_csv(local/'paired_scale_readout/table.csv')
    original = np.load(local/'tokens.npz')
    joint = np.load(local/'spatial_frequency_joint_pooling/features.npz')
    freq = np.load(local/'frequency_conditioned_matching/features.npz')
    snapshots = {int(s['obs_id']): s for s in json.loads(
        (geo.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    model = M3(assets_root, device=device)
    backbone = Backbone(Path(assets_root)/'backbone/resnet50.pth', device=device)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    selected = [int(np.flatnonzero(table.role.eq(role))[0])
                for role in ('utility_train', 'selection', 'evaluation', 'strategy_fit')]
    valid = original['valid'][table.table_index]
    if np.any(~valid[:, 1]):
        selected.append(int(np.flatnonzero(~valid[:, 1])[0]))
    records = []
    for i in sorted(set(selected)):
        up = int(table.iloc[i].table_index)
        prepared = []
        for p, pol in enumerate(('vv', 'vh')):
            if not valid[i, p]:
                prepared.append(None)
                continue
            pixels, footprint, geometry, placement = geo.native(table, snapshots, i, p)
            with Image.open(table.iloc[i]['image_'+pol]) as image:
                raw = np.array(image.convert('L'))
            prepared.append(dict(pixels=pixels, footprint=footprint, geometry=geometry,
                                 placement=placement, raw=raw))
        actual = model.extract(prepared, backbone(prepared))
        expected = dict(layer1=original['layer1'][up], stem=original['stem'][up],
                        mass=original['mass'][up], mean=joint['mean'][i], joint=joint['joint'][i],
                        e_image=freq['e_image'][i], coverage=freq['coverage'][i])
        errors = {key: float(np.abs(actual[key]-value).max()) for key, value in expected.items()}
        for key, error in errors.items():
            assert error < (2e-4 if key in ('layer1', 'stem', 'mean', 'joint') else 1e-6), (i, key, error)
        records.append(dict(up_index=up, obs_id=int(table.iloc[i].obs_id), errors=errors))
        print(json.dumps(records[-1]), flush=True)
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--legacy-root')
    parser.add_argument('--assets-root')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--extract', action='store_true')
    args = parser.parse_args()
    if args.legacy_root:
        if args.extract:
            extraction_parity(args.legacy_root, args.assets_root, args.device)
        else:
            legacy_parity(args.legacy_root, args.assets_root, args.device, args.limit)
    else:
        unittest.main(argv=[sys.argv[0]])
