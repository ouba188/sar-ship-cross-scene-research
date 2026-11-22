"""Small real-architecture training/reload test; synthetic data is not a SAR result."""
import contextlib
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest
from argparse import Namespace
import numpy as np
from PIL import Image
import torch
from sar_port.two_stage.cli import (train_stage1, train_h, evaluate, protocol,
                                     load_checkpoint, restore_encoder, training_reference_mask)


class PipelineTests(unittest.TestCase):
    def test_cross_port_preference_and_same_port_identity_safe_fallback(self):
        query = dict(label='cargo', port='a', vessel_id='q', group_id='qg')
        same = dict(label='cargo', port='a', vessel_id='s', group_id='sg')
        other = dict(label='tanker', port='b', vessel_id='t', group_id='tg')
        mask, count = training_reference_mask([query], [query, same, other])
        self.assertEqual(mask.tolist(), [[False, True, True]])
        self.assertEqual(count, 1)
        cross = dict(label='cargo', port='b', vessel_id='x', group_id='xg')
        mask, count = training_reference_mask([query], [query, same, cross])
        self.assertEqual(mask.tolist(), [[False, False, True]])
        self.assertEqual(count, 0)

    def test_train_freeze_match_and_checkpoint_evaluation(self):
        torch.set_num_threads(1)
        torch.manual_seed(123)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            random = np.random.default_rng(123)
            mask = np.full((64, 64), 255, np.uint8)
            Image.fromarray(mask).save(root/'q.png')
            for port in ('source_a', 'source_b', 'validation', 'target'):
                for label in ('cargo', 'tanker'):
                    identity = f'{port}_{label}'
                    path = root/f'{identity}.png'
                    pixels = random.integers(10, 70, (64, 64), dtype=np.uint8)
                    pixels[28:36, 20:44] += 100
                    Image.fromarray(pixels).save(path)
                    view = dict(image=path.name, alignment_valid=True, background_validated=True,
                                background_mask='q.png', ship_axis=dict(center=[31.5, 31.5],
                                length_px=24, width_px=8, angle=0))
                    rows.append(dict(id=identity, label=label, vessel_id=identity, group_id=identity,
                                     port=port, vv=dict(view), vh=dict(view), leakage_controlled=True))
            manifest = root/'manifest.json'; manifest.write_text(json.dumps(rows), encoding='utf-8')
            config = dict(target_ports=['target'], validation_ports=['validation'],
                          batch_size=4, stage1_epochs=1, base_epochs=1, h_epochs=1,
                          prototypes_per_class=2, loss=dict(hard_gate_after=0, warmup_steps=0, ramp_steps=1))
            split = protocol(rows, config)
            first = root/'stage1.pt'; second = root/'h.pt'
            args = Namespace(manifest=manifest, output=first, checkpoint=None)
            with contextlib.redirect_stdout(io.StringIO()):
                train_stage1(args, config, rows, split, 'cpu')
                args.checkpoint, args.output = first, second
                train_h(args, config, rows, split, 'cpu')
                args.checkpoint, args.output, args.split = second, root/'test.json', 'test'
                evaluate(args, rows, 'cpu')
            saved_first, saved_second = load_checkpoint(first), load_checkpoint(second)
            self.assertEqual(saved_first['history'][0]['losses']['controlled_samples'], 4)
            self.assertGreater(saved_first['history'][0]['losses']['scene_targets'], 0)
            for name, value in saved_first['encoder'].items():
                torch.testing.assert_close(value, saved_second['encoder'][name], rtol=0, atol=0)
            self.assertEqual(len(saved_second['reference_ids']), 4)
            self.assertTrue(all(v.startswith('source_') for v in saved_second['reference_ids']))
            self.assertGreater(saved_second['h_history'][-1]['losses']['match_samples'], 0)
            result = json.loads(args.output.read_text(encoding='utf-8'))
            self.assertEqual(result['count'], 2)
            restored = restore_encoder(saved_second, 'cpu').freeze_for_h()
            self.assertTrue(all(not p.requires_grad for p in restored.parameters()))


if __name__ == '__main__':
    unittest.main()
