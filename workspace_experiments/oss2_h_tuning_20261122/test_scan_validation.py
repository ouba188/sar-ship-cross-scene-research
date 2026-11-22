"""Small CPU/NumPy tests; never import Torch or access any SAR sample."""
import json
from pathlib import Path
import tempfile
import unittest
import scan_validation as scan


class ScanChecks(unittest.TestCase):
    def test_memory_guard_precedes_heavy_imports(self):
        old = scan.memory
        scan.memory = lambda: dict(available_gib=.4, total_gib=16, percent_used=97)
        try:
            with self.assertRaises(scan.MemoryPressure):
                scan.guard_memory(2., 'test')
        finally:
            scan.memory = old

    def test_rescue_harm_and_strict_bound(self):
        import numpy as np
        targets = np.arange(227) % 4
        z0 = np.zeros((227, 4), dtype=np.float32)
        z0[np.arange(227), targets] = 10
        evidence = np.zeros_like(z0)
        z0[0] = (0, 6, -5, -5)      # Cannot rescue at gamma=1; can at gamma=2.
        evidence[0, 0] = 1
        z0[1] = (0, 1, -5, -5)      # Correct h0; harmed at gamma=.25.
        evidence[1, 0] = 1
        z0[2] = (-5, -5, 0, 2)      # E ranks true first, gap .6; gamma=.5 insufficient.
        evidence[2] = (0, 0, .9, .3)
        plan, record = {'fixture': True, 'manifest_sha256': 'fixture', 'seed': 29}, {'sha256': 'fixture'}
        info = dict(plan=plan, checkpoint_sha256='fixture', classes=['0','1','2','3'],
                    selected_epoch=2, selected_validation_metrics={})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'fixture.npz'
            np.savez(path, z0=z0, evidence=evidence, targets=targets,
                     support=np.ones_like(z0, dtype=bool), metadata_json=json.dumps(info))
            result = scan.aggregate_cache(path, plan, record, np)
        by_gamma = {r['gamma']: r for r in result['scan']}
        self.assertEqual(by_gamma[0]['rescued'], 0)
        self.assertEqual(by_gamma[0]['harmed'], 0)
        self.assertEqual(by_gamma[.25]['harmed'], 1)
        self.assertEqual(by_gamma[.5]['rescued'], 0)
        self.assertEqual(by_gamma[1]['rescued'], 1)
        self.assertEqual(by_gamma[1]['h0_wrong_margin_above_bound'], 1)
        self.assertEqual(by_gamma[1]['h0_wrong_evidence_correct_but_above_bound'], 1)
        self.assertEqual(by_gamma[2]['rescued'], 2)
        self.assertTrue(all(r['bound_invariant_violations']==0 for r in result['scan']))
        self.assertEqual(result['checkpoint_selected_epoch'], 2)
        self.assertEqual(result['seed'], 29)
        self.assertEqual(sum(v['count'] for v in result['h0']['per_class'].values()), 227)
        self.assertNotIn('query_ids', result)

    def test_snapshot_is_immutable_and_rejects_different_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'fake_checkpoint.pt'
            source.write_bytes(b'first checkpoint fixture')
            output = root/'seed29'
            output.mkdir()
            pinned, record = scan.snapshot(29, output, source)
            self.assertEqual(pinned.read_bytes(), source.read_bytes())
            self.assertTrue(record['stat_unchanged_during_copy'])
            source.write_bytes(b'new selected checkpoint fixture')
            again, repeated_record = scan.snapshot(29, output, source)
            self.assertEqual(again.read_bytes(), b'first checkpoint fixture')
            self.assertEqual(record, repeated_record)
            another = root/'different_checkpoint.pt'
            another.write_bytes(b'different fixture')
            with self.assertRaisesRegex(RuntimeError, 'another checkpoint source'):
                scan.snapshot(29, output, another)


if __name__ == '__main__':
    scan.guard_memory(.25, 'NumPy-only fixture')
    unittest.main()
