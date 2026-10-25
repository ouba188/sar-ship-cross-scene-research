"""Unit checks plus opt-in parity against the authoritative rebuilt artifacts.

Set M123_LEGACY_ROOT to model_closeout_20261023 to enable migration checks.
No legacy module is imported by the standalone inference package.
"""
import ast
import csv
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from sar_m123 import regions
from sar_m123.m12 import M12, SummaryReader, attention
from tools.export_m12 import export


def original_functions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(nodes) != len(names):
        raise AssertionError('Reference functions missing from original source.')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


class RegionTests(unittest.TestCase):
    def test_pixel_boundary_and_mass(self):
        x = np.zeros((160, 160)); x[0, 2] = 1
        mass = regions.cell_mass(x)
        self.assertEqual(mass[0, 0], .5)
        self.assertEqual(mass[0, 1], .5)
        self.assertEqual(mass.sum(), 1.)
        g = np.array([73.2, 81.7, 27., 9., .37, .8])
        m = regions.soft_regions(g)
        np.testing.assert_allclose(regions.cell_mass(m).sum((-2, -1)), m.sum((-2, -1)), atol=1e-10)
        for k in range(4):
            np.testing.assert_allclose(np.rot90(m, k, (-2, -1)),
                                      regions.soft_regions(regions.rotate_geometry(g, k)), atol=1e-12)

    def test_missing_geometry_and_empty_attention(self):
        m = regions.soft_regions(None)
        self.assertFalse(m.any())
        q = regions.region_quality(m, np.ones((160, 160)), None)
        self.assertFalse(q['min_neff'].any())
        packed = attention(torch.randn(1, 8, 100, dtype=torch.float64),
                           torch.zeros(1, 3, 100, dtype=torch.float64), torch.tensor(1.))
        self.assertTrue(torch.isfinite(packed).all())


@unittest.skipUnless(os.environ.get('M123_LEGACY_ROOT'), 'Set M123_LEGACY_ROOT for original-asset parity.')
class LegacyParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.legacy = Path(os.environ['M123_LEGACY_ROOT'])
        cls.up = cls.legacy/'regional_utility_pilot'/'upstream'
        cls.temp = tempfile.TemporaryDirectory()
        cls.assets = Path(cls.temp.name)
        cls.exported = export(cls.legacy, cls.assets)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_all_saved_head_scores_and_selected_zero_steps(self):
        with np.load(self.up/'summaries.npz') as z:
            s = {k: z[k] for k in ('mean3', 'hb3', 'summary1', 'quality', 'geometry', 'valid')}
        epochs = {17: (0, 15), 29: (70, 0), 43: (10, 0)}
        for seed in epochs:
            model = M12(self.assets, seed)
            with np.load(self.up/f'base_seed{seed}.npz') as z:
                base = z['B']
            with np.load(self.up/f'strategies_seed{seed}.npz') as z:
                expected = z['logits']
            score, details = model(base, s)
            np.testing.assert_allclose(details['s0'], expected[:, 0], atol=3e-6, rtol=0)
            np.testing.assert_allclose(score, expected[:, 2], atol=3e-6, rtol=0)
            np.testing.assert_array_equal(score.argmax(1), expected[:, 2].argmax(1))
            self.assertEqual(tuple(details['selected_steps'].values()), epochs[seed])
            if seed == 17:
                self.assertFalse(details['delta_m1'].any())
            else:
                self.assertFalse(details['delta_m2'].any())
            single, _ = model(base[0], {k: v[0] for k, v in s.items()})
            np.testing.assert_allclose(single, expected[0, 2], atol=3e-6, rtol=0)
            print('head parity', seed, len(base), float(np.max(np.abs(score-expected[:, 2]))))
        self.assertEqual(len(self.exported), 9)

    def test_summary_against_original_batched_operators(self):
        # Original functions are executed only in this migration test, without
        # importing their data loaders, historical paths, or training entrypoints.
        geo_path = self.legacy.parent/'eight_class_evidence_20261023'/'obb_dynamic_readout'/'common.py'
        geo = original_functions(geo_path,
            ['soft_regions', 'planned_mass', 'cell_mass', 'region_quality'],
            dict(np=np, SIZE=160, EDGES=regions.EDGES.copy(), YY=regions.YY.copy(), XX=regions.XX.copy()))
        original = original_functions(self.up/'extract_source.py',
                                      ['centered_response', 'attention'], dict(torch=torch))
        reader = SummaryReader(self.assets)
        torch.manual_seed(81)
        prepared, maps = [], []
        for p in range(2):
            fp = np.zeros((160, 160), np.float32)
            fp[11:148, 18:143] = 1
            g = np.array([76.3, 82.1, 41., 13., .71+p*.31, .83])
            prepared.append(dict(footprint=fp, geometry=g))
            maps.append({layer: torch.rand(4, d, side, side)
                         for layer, d, side in [('layer1', 256, 40), ('layer3', 1024, 10)]})
        out = reader.summarize(prepared, maps)
        expected_mean = np.zeros((2, 8)); expected_hb = np.zeros((2, 8, 2))
        expected_regions = np.zeros((2, 3, 8, 3))
        masks = [geo['soft_regions'](v['geometry']) for v in prepared]
        flags = []
        for p, item in enumerate(prepared):
            q = geo['region_quality'](masks[p], item['footprint'], item['geometry'])
            flags.append((q['area'] > 0) & (q['min_neff'] >= 4))
            np.testing.assert_array_equal(out['quality'][p],
                np.stack([q['area'], q['coverage'], q['min_neff'], flags[-1]], -1))
        for k in range(4):
            for layer in ('layer1', 'layer3'):
                z = torch.stack([m[layer][k] for m in maps]).flatten(2).double()
                W = torch.tensor(reader.readers[layer])
                response = torch.einsum('bck,bku->bcu', W, z)
                if layer == 'layer3':
                    weights = np.stack([np.rot90(v['footprint'], k).reshape(10, 16, 10, 16).mean((1, 3))
                                        for v in prepared]).reshape(2, 1, -1)
                else:
                    weights = np.stack([geo['cell_mass'](np.rot90(masks[p]*v['footprint'], k, (-2, -1)))
                                        for p, v in enumerate(prepared)]).reshape(2, 3, -1)
                tau = torch.tensor(reader.tau[layer], dtype=torch.float64)[:, None, None, None]
                packed = original['attention'](response, torch.tensor(weights, dtype=torch.float64),
                                               tau, layer == 'layer3').numpy()
                if layer == 'layer3':
                    expected_mean += packed[:, 0, :, 0]/4
                    expected_hb += packed[:, 0, :, 1:]/4
                else:
                    for p in range(2):
                        packed[p, ~flags[p]] = 0
                    expected_regions += packed/4
        for key, expected in [('mean3', expected_mean), ('hb3', expected_hb), ('summary1', expected_regions)]:
            np.testing.assert_allclose(out[key], expected, atol=1e-10, rtol=1e-12)
        missing = reader.summarize([prepared[0], None], [maps[0], None])
        for key in ('mean3', 'hb3', 'summary1', 'quality', 'geometry', 'valid'):
            self.assertFalse(missing[key][1].any())
        no_geometry = dict(prepared[0], geometry=None)
        empty = reader.summarize([no_geometry, None], [maps[0], None])
        self.assertFalse(empty['summary1'].any())
        self.assertFalse(empty['geometry'].any())
        self.assertTrue(empty['valid'][0])

    @unittest.skipUnless(os.environ.get('M123_REAL_CHIPS') == '1',
                         'Set M123_REAL_CHIPS=1 for real PNG/GPU/cache validation.')
    def test_real_chips_gpu_against_saved_summaries(self):
        from sar_m123.input import prepare
        from sar_m123.backbone import Backbone

        if not torch.cuda.is_available():
            self.skipTest('CUDA is required for the original extraction parity check.')
        with (self.up/'table.csv').open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        snapshot_path = (self.legacy.parents[1]/'same_vessel_gallery_20261019'/
                         'local_condition_module'/'view_snapshot.json')
        snapshots = {int(s['obs_id']): s for s in json.loads(snapshot_path.read_text(encoding='utf-8'))}
        with np.load(self.up/'summaries.npz') as z:
            expected = {k: z[k] for k in ('mean3', 'hb3', 'summary1', 'quality', 'geometry', 'valid')}
        missing = np.flatnonzero(~expected['valid'][:, 1])
        ids = list(dict.fromkeys([0, 1, 2, 7, 17, 43, 100] + missing[:1].tolist()))
        weights = Path(__file__).resolve().parents[1]/'assets'/'backbone'/'resnet50.pth'
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        backbone = Backbone(weights, device='cuda')
        reader = SummaryReader(self.assets)
        collected = []
        for i in ids:
            row = rows[i]; snapshot = snapshots[int(row['obs_id'])]
            record = {}
            for p, pol in enumerate(('vv', 'vh')):
                if not expected['valid'][i, p]:
                    record[pol] = None
                    continue
                prior = snapshot['images'][pol]['priors']
                record[pol] = dict(image=row['image_'+pol], detections=prior['detections'],
                                   target_index=prior['target_index'], ship_axis=prior.get('ship_axis'),
                                   radar_axis_rad=prior.get('radar_axis'))
            prepared = prepare(record)
            maps = backbone(prepared)
            actual = reader.summarize(prepared, maps)
            errors = {}
            for key, reference in expected.items():
                if key == 'valid':
                    np.testing.assert_array_equal(actual[key], reference[i])
                else:
                    errors[key] = float(np.max(np.abs(actual[key]-reference[i])))
                    tolerance = 1e-8 if key in ('quality', 'geometry') else 1e-4
                    np.testing.assert_allclose(actual[key], reference[i], atol=tolerance, rtol=0,
                                               err_msg=f'obs={row["obs_id"]} {key}')
            collected.append(actual)
            print('real chip parity', row['obs_id'], 'valid', actual['valid'].tolist(), errors)
        batch = {k: np.stack([s[k] for s in collected]) for k in expected}
        for seed in (17, 29, 43):
            with np.load(self.up/f'base_seed{seed}.npz') as z:
                base = z['B'][ids]
            with np.load(self.up/f'strategies_seed{seed}.npz') as z:
                reference = z['logits'][ids, 2]
            scores, _ = M12(self.assets, seed)(base, batch)
            np.testing.assert_allclose(scores, reference, atol=1e-4, rtol=0)
            np.testing.assert_array_equal(scores.argmax(1), reference.argmax(1))
            print('real chip M12 parity', seed, 'max error', float(np.max(np.abs(scores-reference))))


if __name__ == '__main__':
    unittest.main()
