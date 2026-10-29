"""Run with python -B -m unittest tests_cpu -v; standard library only."""
import ast
import importlib.abc
import math
import sys
import unittest

from protocol import HERE, CLASS_INDICES, assert_rows, class_weights4
from converter import corners, view_spec


class NoTensorImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'numpy', 'sar_m123'):
            raise AssertionError(f'CPU preparation imported {fullname}')


class PreparationTests(unittest.TestCase):
    def test_imports_and_help_surface_are_lightweight(self):
        blocker = NoTensorImports()
        sys.meta_path.insert(0, blocker)
        try:
            import run
            import checkreadiness
            import tests_gpu
            self.assertTrue(callable(run.main) and callable(checkreadiness.check) and callable(tests_gpu.check_gpu))
        finally:
            sys.meta_path.remove(blocker)
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('numpy', sys.modules)

    def test_python_syntax_without_importing_runtime(self):
        for path in HERE.glob('*.py'):
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))

    def test_raw_width_height_angle_not_sorted_axis(self):
        raw = [10., 20., 2., 8., math.pi/6]
        pts = corners(raw)
        dx, dy = pts[1][0]-pts[0][0], pts[1][1]-pts[0][1]
        self.assertAlmostEqual(math.hypot(dx, dy), 2.)
        self.assertAlmostEqual(math.atan2(dy, dx), math.pi/6)
        self.assertAlmostEqual(math.dist(pts[1], pts[2]), 8.)
        self.assertAlmostEqual(sum(p[0] for p in pts)/4, 10.)
        self.assertAlmostEqual(sum(p[1] for p in pts)/4, 20.)

    def test_invalid_polarization_and_axis_are_not_inferred_from_file(self):
        view = dict(valid_target='True', prepare_ok='True', detected='True', confidence='.8',
            raw_xywhr='[10,20,2,8,0.3]', axis_valid='False', path='existing.png',
            sample_id='q', pol='vv', cx='10', cy='20', length='8', width='2', ship_axis='1.87')
        spec = view_spec(view)
        self.assertIsNone(spec['ship_axis'])
        self.assertIsNone(spec['radar_axis_rad'])
        self.assertEqual(spec['target_index'], 0)
        self.assertEqual(spec['detections'][0]['corners'], corners([10, 20, 2, 8, .3]))
        self.assertIsNone(view_spec(dict(view, valid_target='False')))
        self.assertIsNone(view_spec(dict(view, confidence='.24')))
        self.assertIsNone(view_spec(dict(view, prepare_ok='False')))
        self.assertEqual(view_spec(dict(view, axis_valid='True'))['ship_axis']['length_px'], 8)

    def test_classweights_four_classes_and_repeat_compensation(self):
        records = [dict(id=str(i), label=y, original_class_id=CLASS_INDICES[y],
                        mmsi=ship, split='source_fit')
                   for i, (y, ship) in enumerate(((0, 'a'), (0, 'a'), (0, 'b'),
                                                  (1, 'c'), (2, 'd'), (3, 'e')))]
        self.assertEqual(class_weights4(records, 'fit'), [.375, .375, .75, 1.5, 1.5, 1.5])
        adapt = [dict(r, split='source_adapt') for r in records]
        self.assertEqual(class_weights4(adapt, 'adapt'), [.5, .5, .5, 1.5, 1.5, 1.5])
        with self.assertRaises(ValueError):
            assert_rows([dict(records[0], original_class_id=1)])
        with self.assertRaises(ValueError):
            assert_rows([dict(records[0], label=4)])
        with self.assertRaises(ValueError):
            assert_rows(records, 'source_adapt')


if __name__ == '__main__':
    unittest.main()
