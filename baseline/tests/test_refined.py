import unittest

import numpy as np
import pandas as pd
from scipy import sparse

from baseline.aggregates import ManualAggregates
from baseline.refined import RefinedFeatures
from baseline.run_refined import feature_sets
import test_aggregates


class RefinedFeatureTests(unittest.TestCase):
    def setUp(self):
        fixture = test_aggregates.ManualAggregateTests(); fixture.setUp()
        self.fixture = fixture
        self.ctx = fixture.context()

    def test_sources_match_expansion_membership(self):
        original = pd.DataFrame({'user_id':[1], 'item_id':[30]})
        expanded = self.ctx.expand_candidates(original)
        result = RefinedFeatures(self.ctx, original).features(expanded)
        self.assertTrue((result.ref_source_count >= 1).all())
        user1 = expanded.user_id.eq(1)
        new = expanded.item_id.eq(20) & user1
        self.assertEqual(result.loc[new, 'ref_source_co'].item(), 1)
        self.assertEqual(result.loc[new, 'ref_source_original'].item(), 0)
        self.assertEqual(result.loc[user1 & expanded.item_id.eq(30), 'ref_source_original'].item(), 1)
        self.assertTrue(np.isfinite(result.to_numpy()).all())

    def test_local_interest_matches_hand_calculation_and_same_day_pooling(self):
        ctx = self.ctx
        # User 2 has seeds 10 and 20 on the same date; candidate 30 has links .2 and .6.
        ctx.graph = sparse.csr_matrix(np.array([[0,0,.2,0],[0,0,.6,0],[0,0,0,0],[0,0,0,0]], dtype='float32'))
        frame = pd.DataFrame({'user_id':[2,2], 'item_id':[30,40]})
        features = RefinedFeatures(ctx, frame.iloc[:0]).features(frame)
        self.assertAlmostEqual(features.ref_interest_top3_mean.iloc[0], .4, places=6)
        self.assertAlmostEqual(features.ref_interest_top1_fraction.iloc[0], .75, places=6)
        self.assertAlmostEqual(features.ref_interest_weighted_age.iloc[0], 1, places=6)
        self.assertAlmostEqual(features.ref_interest_last_day_mean.iloc[0], .4, places=6)
        self.assertEqual(features.ref_interest_weighted_age.iloc[1], 999)
        self.assertEqual(features.ref_interest_top1_fraction.iloc[1], 0)

    def test_lift_uses_global_prior_and_shrinkage(self):
        frame = pd.DataFrame({'user_id':[1], 'item_id':[20]})
        features = RefinedFeatures(self.ctx, frame.iloc[:0]).features(frame)
        # Historical actor masses: A=2, B=5. Personal mass for B=.5, total mass=1.
        expected = np.log((.5 + 10 * 5/7) / (11 * 5/7))
        self.assertAlmostEqual(features.ref_lift_actors_mean.iloc[0], expected, places=6)
        self.assertEqual(features.ref_lift_actors_support.iloc[0], .5)

    def test_seven_day_boundaries_match_date_filter(self):
        cutoff = self.fixture.cutoff
        dates = pd.date_range(cutoff-pd.Timedelta(days=9), cutoff-pd.Timedelta(days=1))
        np.testing.assert_array_equal((cutoff-dates).days <= 7,
                                      dates >= cutoff-pd.Timedelta(days=7))

    def test_same_day_order_does_not_change_new_features(self):
        frame = pd.DataFrame({'user_id':[2,2], 'item_id':[30,40]})
        original = frame.iloc[:0]
        first = RefinedFeatures(self.ctx, original).features(frame)
        seeds, ages = self.ctx.histories[2]
        self.ctx.histories[2] = (seeds[::-1], ages[::-1])
        second = RefinedFeatures(self.ctx, original).features(frame)
        np.testing.assert_allclose(first.to_numpy(), second.to_numpy(), atol=1e-6)

    def test_ablation_groups_preserve_baseline_and_separate_new_features(self):
        baseline = ['age','manual_user_history_size','manual_co_sum']
        columns = baseline + ['manual_v3_x','ref_source_x','ref_interest_x','ref_lift_x']
        groups = feature_sets(columns, baseline)
        self.assertEqual(groups['v2_gpu'], baseline)
        self.assertNotIn('manual_user_history_size', groups['clean'])
        self.assertNotIn('ref_lift_x', groups['sources'])
        self.assertIn('ref_lift_x', groups['combined'])
        self.assertNotIn('manual_v3_x', groups['combined'])


if __name__ == '__main__':
    unittest.main()
