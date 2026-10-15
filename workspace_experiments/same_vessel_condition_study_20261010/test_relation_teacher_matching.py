import unittest

import numpy as np
import pandas as pd

from relation_teacher_matching import MATCH_COLUMNS, build_matches, summarize_matches


def observation(mmsi, product, port='p', class_id=0, incidence_deg=30.,
                pass_name='ASCENDING', platform='S1A'):
    return dict(mmsi=mmsi, product=product, port=port, class_id=class_id,
                incidence_deg=incidence_deg, pass_name=pass_name, platform=platform)


class RelationTeacherMatchingTests(unittest.TestCase):
    def test_excludes_anchor_scene_same_ship_and_other_class(self):
        meta = pd.DataFrame([
            observation(1, 'a'), observation(1, 'a'), observation(1, 'b'),
            observation(2, 'a'), observation(3, 'b', class_id=1),
            observation(4, 'c'),
        ])
        matches = build_matches(meta)
        first = matches[matches.anchor == 0].iloc[0]
        self.assertEqual(first.same_indices, (2,))
        self.assertEqual(first.other_indices, (5,))
        for pair in matches.itertuples(index=False):
            anchor = meta.iloc[pair.anchor]
            same, other = meta.iloc[list(pair.same_indices)], meta.iloc[list(pair.other_indices)]
            self.assertTrue((same.mmsi == anchor.mmsi).all())
            self.assertTrue((other.mmsi != anchor.mmsi).all())
            self.assertTrue((other.class_id == anchor.class_id).all())
            self.assertTrue((same['product'] != anchor['product']).all())
            self.assertTrue((other['product'] != anchor['product']).all())

    def test_same_scene_precedes_closer_fallback_and_requires_same_port(self):
        meta = pd.DataFrame([
            observation(1, 'a'), observation(1, 'b'),
            observation(2, 'c'), observation(3, 'b', port='elsewhere'),
            observation(4, 'b', incidence_deg=31., pass_name=None, platform=None),
        ])
        pair = build_matches(meta).query('anchor == 0').iloc[0]
        self.assertEqual(pair.other_indices, (4,))
        self.assertEqual(pair.match_kind, 'same_scene')
        self.assertEqual(pair.incidence_gap, 1.)
        meta.loc[4, 'incidence_deg'] = 35.
        pair = build_matches(meta).query('anchor == 0').iloc[0]
        self.assertEqual(pair.match_kind, 'fallback')
        self.assertEqual(pair.other_indices, (2,))
        meta.loc[1, 'incidence_deg'] = np.nan
        pair = build_matches(meta).query('anchor == 0').iloc[0]
        self.assertEqual(pair.match_kind, 'same_scene')
        self.assertTrue(np.isnan(pair.incidence_gap))

    def test_fallback_requires_observations_conditions_and_two_degree_limit(self):
        base = pd.DataFrame([
            observation(1, 'a', port='anchor_port', incidence_deg=70.),
            observation(1, 'b'), observation(2, 'c', incidence_deg=32.),
        ])
        pair = build_matches(base).query('anchor == 0').iloc[0]
        self.assertEqual((pair.match_kind, pair.incidence_gap), ('fallback', 2.))
        for position in (1, 2):
            for column, value in [('incidence_deg', np.nan), ('incidence_deg', np.inf),
                                  ('pass_name', None), ('pass_name', ''),
                                  ('platform', None), ('platform', 'unknown')]:
                with self.subTest(position=position, column=column, value=value):
                    changed = base.copy()
                    changed.loc[position, column] = value
                    self.assertTrue(build_matches(changed).query('anchor == 0').empty)
        for column, value in [('incidence_deg', 32.001), ('port', 'other_port'),
                              ('pass_name', 'DESCENDING'), ('platform', 'S1B')]:
            with self.subTest(column=column, value=value):
                changed = base.copy()
                changed.loc[2, column] = value
                self.assertTrue(build_matches(changed).query('anchor == 0').empty)

    def test_duplicate_chips_remain_one_scene_and_unmatched_scenes_are_omitted(self):
        meta = pd.DataFrame([
            observation(1, 'a'), observation(1, 'b'), observation(1, 'b'),
            observation(1, 'd', incidence_deg=50.),
            observation(2, 'b'), observation(2, 'b'),
            observation(1, 'e', incidence_deg=80., port='unmatched'),
            observation(3, 'd', incidence_deg=50.),
        ])
        pairs = build_matches(meta).query('anchor == 0')
        self.assertEqual(pairs.same_indices.tolist(), [(1, 2), (3,)])
        self.assertEqual(pairs.other_indices.tolist(), [(4, 5), (7,)])
        values = np.array([0., 2., 4., 10., 5., 7., 999., 14.])
        same_mean = np.mean([values[list(indices)].mean() for indices in pairs.same_indices])
        self.assertEqual(same_mean, 6.5)

    def test_deterministic_positional_indices_and_fallback_tiebreaks(self):
        meta = pd.DataFrame([
            observation(1, 'a'), observation(1, 'b'),
            observation(9, 'x', incidence_deg=30.5),
            observation(2, 'z', incidence_deg=31.),
            observation(2, 'y', incidence_deg=29.),
        ], index=[90, 90, -4, 8, 600])
        original = meta.copy(deep=True)
        pairs = build_matches(meta)
        self.assertEqual(pairs.query('anchor == 0').iloc[0].other_indices, (2,))
        pd.testing.assert_frame_equal(pairs, build_matches(meta.reset_index(drop=True)))
        pd.testing.assert_frame_equal(meta, original)
        meta.iloc[2, meta.columns.get_loc('incidence_deg')] = 32.
        self.assertEqual(build_matches(meta).query('anchor == 0').iloc[0].other_indices, (4,))
        for position in (2, 3, 4):
            meta.iloc[position, meta.columns.get_loc('product')] = 'b'
        self.assertEqual(build_matches(meta).query('anchor == 0').iloc[0].other_indices, (3, 4))

    def test_metadata_audit_and_empty_frame_schema(self):
        meta = pd.DataFrame([
            observation(1, 'a'), observation(1, 'b'), observation(2, 'b'),
            observation(4, 'c'),
            observation(3, 'x', class_id=1),
        ])
        matches = build_matches(meta)
        audit = summarize_matches(meta, matches)
        self.assertEqual((audit['unique_anchors'], audit['unique_ships'],
                          audit['unique_classes'], audit['match_count']), (2, 1, 1, 2))
        self.assertEqual(audit['per_class']['0']['anchor_coverage'], .5)
        self.assertEqual(audit['per_class']['1']['unique_anchors'], 0)
        self.assertEqual(audit['same_scene_fraction'], .5)
        self.assertEqual(audit['fallback_fraction'], .5)
        self.assertEqual(audit['incidence_gap'], dict(count=2, mean=0., median=0., max=0.))
        for frame in (meta.iloc[4:], meta.iloc[:0]):
            with self.subTest(rows=len(frame)):
                empty = build_matches(frame)
                self.assertTrue(empty.empty)
                self.assertEqual(list(empty.columns), MATCH_COLUMNS)
                self.assertEqual(empty.anchor.dtype, np.dtype('int64'))
                self.assertEqual(empty.incidence_gap.dtype, np.dtype('float64'))
                summary = summarize_matches(frame, empty)
                self.assertEqual(summary['unique_anchors'], 0)
                self.assertEqual(summary['same_scene_fraction'], 0.)
                self.assertIsNone(summary['incidence_gap']['mean'])


if __name__ == '__main__':
    unittest.main()
