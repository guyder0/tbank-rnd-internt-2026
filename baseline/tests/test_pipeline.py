"""Temporal and ranking invariants; run with unittest discovery."""
import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from baseline.pipeline import evaluate, history_before, make_candidates, make_features, score_model, targets


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.cutoff = pd.Timestamp('2021-08-01')
        self.events = pd.DataFrame({
            'user_id': [1, 2, 2, 1], 'item_id': [10, 20, 30, 20],
            'date': pd.to_datetime(['2021-07-31', '2021-07-30', '2021-07-01', '2021-08-01']),
        })
        self.items = pd.DataFrame({
            'item_id': [10, 20, 30, 40], 'genres': ['драма', 'драма', 'комедия', 'драма'],
            'countries': ['Россия'] * 4, 'directors': ['А', 'А', 'Б', 'В'],
            'content_type': ['film'] * 4, 'release_year': [2020] * 4,
            'age_rating': [16] * 4, 'for_kids': [0] * 4,
        })
        self.users = pd.DataFrame({'user_id': [1, 2], 'age': ['age_25_34'] * 2,
                                   'income': ['unknown'] * 2, 'sex': ['М', 'Ж'], 'kids_flg': [0, 1]})
        self.config = {'popular_candidates': 10, 'genre_candidates': 6}

    def test_cutoff_is_exclusive_and_labels_include_boundary(self):
        past = history_before(self.events, self.cutoff)
        self.assertEqual(len(past), 3)
        self.assertEqual(targets(self.events, [1], self.cutoff, 7), {1: {20}})

    def test_candidate_exclusions_and_future_invariance(self):
        past = history_before(self.events, self.cutoff)
        first = make_candidates(past, [1], self.items, self.config, self.cutoff)
        mutated = pd.concat([self.events, pd.DataFrame({'user_id': [1], 'item_id': [40],
                                                       'date': [self.cutoff + pd.Timedelta(days=1)]})])
        second = make_candidates(history_before(mutated, self.cutoff), [1], self.items, self.config, self.cutoff)
        pd.testing.assert_frame_equal(first, second)
        self.assertNotIn(10, first.item_id.tolist())
        self.assertNotIn(40, first.item_id.tolist())

    def test_features_are_manual_past_aggregates(self):
        candidates = pd.DataFrame({'user_id': [1, 1], 'item_id': [20, 30]})
        history = history_before(self.events, self.cutoff)
        result = make_features(candidates, history, self.users, self.items, self.cutoff, [7, 30, 90])
        self.assertEqual(result.loc[0, 'user_count_7d'], 1)
        self.assertEqual(result.loc[0, 'affinity_genres_7d_sum'], 1)
        self.assertEqual(result.loc[1, 'affinity_genres_7d_sum'], 0)
        self.assertEqual(result.loc[0, 'user_last_days'], 1)
        with self.assertRaises(ValueError):
            make_features(candidates, self.events, self.users, self.items, self.cutoff, [7, 30, 90])

    def test_metrics_keep_users_with_no_retrieved_positives(self):
        frame = pd.DataFrame({'user_id': [1, 1], 'item_id': [10, 20]})
        result, _ = evaluate(frame, [2, 1], {1: {10}, 2: {30}}, k=10)
        self.assertEqual(result['users'], 2)
        self.assertAlmostEqual(result['ndcg@10'], 0.5)
        self.assertAlmostEqual(result['candidate_recall'], 0.5)
        self.assertAlmostEqual(result['map@10'], 0.5)

    def test_ranking_ties_are_deterministic(self):
        frame = pd.DataFrame({'user_id': [1, 1], 'item_id': [20, 10]})
        result, _ = evaluate(frame, np.zeros(2), {1: {10}}, k=1)
        self.assertEqual(result['ndcg@1'], 1)

    def test_saved_split_scoring_and_bootstrap(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            frame = pd.DataFrame({'user_id': [1, 1], 'item_id': [10, 20],
                                  'query_id': [1, 1], 'item_count_30d': [2, 1]})
            frame.to_pickle(output / 'validation.pkl')
            (output / 'truth.json').write_text(json.dumps({'1': [10]}))
            manifest = {'config': {'seed': 42}, 'splits': [
                {'split': 'validation', 'path': 'validation.pkl', 'truth': 'truth.json'}]}
            model = SimpleNamespace(predict=lambda features: np.array([2, 1]))
            report = {}
            score_model(SimpleNamespace(output=output), model, ['item_count_30d'], manifest, report, 'validation')
            self.assertEqual(report['validation']['ndcg_delta_95pct_ci'], [0, 0])
            self.assertEqual(report['validation']['catboost']['ndcg@10'], 1)
            self.assertTrue((output / 'validation_recommendations.csv').exists())


if __name__ == '__main__':
    unittest.main()
