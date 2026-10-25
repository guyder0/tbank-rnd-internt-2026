import unittest
import tempfile
import json
from pathlib import Path

import numpy as np
import pandas as pd

from baseline.aggregates import ManualAggregates
from baseline.pipeline import evaluate, make_candidates, make_features
from baseline.strong import ndcg_fast, reserve, final, select_features
import test_pipeline


class ManualAggregateTests(unittest.TestCase):
    def setUp(self):
        fixture = test_pipeline.PipelineTests(); fixture.setUp()
        self.items = fixture.items.assign(actors=['А, Б','Б','В','Г'], studios=['С','С','Т','У'])
        self.users = fixture.users
        self.cutoff = fixture.cutoff
        self.history = pd.DataFrame({'user_id':[1,2,3,4,2,3,4],
                                    'item_id':[10,10,10,10,20,20,20],
                                    'date':pd.to_datetime(['2021-07-31']*7)})
        self.config = {'graph_history_limit':100,'neighbors':3,'similarity_shrinkage':10,
                       'minimum_support':1,'co_candidates':3,'fresh_candidates':3,'cohort_candidates':3}

    def context(self, history=None):
        return ManualAggregates(self.history if history is None else history,self.users,self.items,
                                self.cutoff,self.config,user_ids=[1,2])

    def test_cooccurrence_matches_hand_calculation(self):
        ctx = self.context()
        # 3 users consumed both, 4 consumed item 10, 3 consumed item 20.
        expected = 3 / np.sqrt(4*3) * 3/(3+10)
        self.assertAlmostEqual(float(ctx.graph[0,1]),expected,places=6)
        self.assertTrue(np.all(ctx.graph.diagonal()==0))
        frame = pd.DataFrame({'user_id':[1,1],'item_id':[20,30]})
        features = ctx.features(frame)
        self.assertAlmostEqual(features.manual_co_sum.iloc[0],expected,places=6)
        self.assertEqual(features.manual_co_sum.iloc[1],0)
        self.assertEqual(features.manual_actors_last1_overlap.iloc[0],1)
        self.assertFalse(features.isna().any().any())

    def test_extended_token_probabilities_and_graph_recency(self):
        config = {**self.config, 'extended_aggregates': True}
        ctx = ManualAggregates(self.history, self.users, self.items, self.cutoff,
                               config, user_ids=[1, 2])
        features = ctx.features(pd.DataFrame({'user_id':[1,1], 'item_id':[20,30]}))
        # User 1 saw actors A and B in one item: each receives mass 1/2.
        self.assertAlmostEqual(features.manual_v3_actors_entropy.iloc[0], np.log(2), places=6)
        self.assertAlmostEqual(features.manual_v3_actors_concentration.iloc[0], .5)
        self.assertAlmostEqual(features.manual_v3_actors_probability.iloc[0], .5)
        self.assertAlmostEqual(features.manual_v3_actors_decay7_probability.iloc[0], .5)
        self.assertEqual(features.manual_v3_co_nearest_age.tolist(), [1, 999])
        self.assertEqual(features.manual_v3_co_support_fraction.tolist(), [1, 0])
        self.assertEqual(features.manual_v3_item_observed_age.tolist(), [1, 999])
        self.assertTrue(np.isfinite(features.to_numpy()).all())
        # Prefix cutoff invariance even when all later outcomes are replaced.
        future = self.history.copy()
        future['date'] = self.cutoff + pd.Timedelta(days=1)
        combined = pd.concat([self.history, future])
        other = ManualAggregates(combined[combined.date < self.cutoff], self.users,
                                self.items, self.cutoff, config, user_ids=[1,2])
        pd.testing.assert_frame_equal(features, other.features(
            pd.DataFrame({'user_id':[1,1], 'item_id':[20,30]})))

    def test_labels_and_outcome_columns_do_not_affect_features(self):
        ctx = self.context()
        candidate = pd.DataFrame({'user_id':[1,1], 'item_id':[20,30], 'label':[1,0]})
        before = make_features(candidate, self.history, self.users, self.items,
                               self.cutoff, [7,30,90]).drop(columns='label')
        changed = candidate.assign(label=[0,1])
        after = make_features(changed, self.history, self.users, self.items,
                              self.cutoff, [7,30,90]).drop(columns='label')
        pd.testing.assert_frame_equal(before, after)
        poisoned = self.history.assign(label=1, total_dur=1e12, watched_pct=100)
        other = self.context(poisoned)
        pd.testing.assert_frame_equal(ctx.features(candidate), other.features(changed))

    def test_explicit_feature_list_cannot_include_labels_or_ids(self):
        frame = pd.DataFrame(columns=['user_id','item_id','label','query_id','age','manual_co_sum'])
        for forbidden in ['label','user_id','item_id','query_id']:
            with self.assertRaises(ValueError):
                select_features(frame, {'feature_columns':['age',forbidden]})
        self.assertEqual(select_features(frame, {}), ['age','manual_co_sum'])
        self.assertEqual(select_features(frame, {'feature_columns':['age']}), ['age'])

    def test_future_is_rejected(self):
        future = self.history.copy();future.loc[0,'date']=self.cutoff
        with self.assertRaises(ValueError): self.context(future)

    def test_expansion_excludes_seen_and_never_observed_items(self):
        ctx=self.context()
        original=pd.DataFrame({'user_id':[1],'item_id':[30]})
        expanded=ctx.expand_candidates(original)
        user1=set(expanded.loc[expanded.user_id==1,'item_id'])
        self.assertNotIn(10,user1)
        self.assertNotIn(40,user1)
        self.assertIn(20,user1)
        # A user with an empty original list still receives past-only candidates.
        self.assertIn(1,ctx.expand_candidates(original.iloc[:0]).user_id.tolist())

    def test_cohort_smoothing_matches_formula(self):
        ctx=self.context()
        features=ctx.features(pd.DataFrame({'user_id':[1],'item_id':[20]}))
        prior=3/7
        expected=(1+100*prior)/(3+100) # users 1 and 2 share age: 3 historical pairs.
        self.assertAlmostEqual(features.manual_cohort_age_30d_probability.iloc[0],expected,places=6)

    def test_fast_ndcg_matches_full_metric_with_retrieval_misses(self):
        frame=pd.DataFrame({'user_id':[1,1,2,2],'item_id':[10,20,10,30],'label':[1,0,0,1]})
        truth={1:{10,40},2:{30},3:{50}}
        scores=np.array([1,2,0,3])
        expected,_=evaluate(frame,scores,truth)
        self.assertAlmostEqual(ndcg_fast(frame,scores,truth),expected['ndcg@10'])

    def test_positive_activity_filter_does_not_change_user_candidates(self):
        config={'popular_candidates':10,'genre_candidates':6}
        all_users=make_candidates(self.history,[1,2],self.items,config,self.cutoff)
        just_one=make_candidates(self.history,[1],self.items,config,self.cutoff)
        pd.testing.assert_frame_equal(all_users[all_users.user_id==1].reset_index(drop=True),just_one)

    def test_graph_cap_preserves_full_personal_history(self):
        config = {**self.config, 'graph_history_limit': 1}
        ctx = ManualAggregates(self.history,self.users,self.items,self.cutoff,config,user_ids=[2])
        self.assertEqual(len(ctx.histories[2][0]),2)
        self.assertEqual(ctx.graph.nnz,0)

    def test_reservation_needs_no_outcomes_and_detects_modified_partition(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);base=root/'base';output=root/'strong'
            base.mkdir();output.mkdir()
            pd.Series([1],name='user_id').to_csv(base/'sample_users.csv',index=False)
            config={'holdout_seed':42,'holdout_users':2}
            # Deliberately no timestamp, item or label columns.
            population=pd.DataFrame({'user_id':[1,2,3,4]})
            selected=reserve(output,base,config,population)
            self.assertNotIn(1,selected)
            np.testing.assert_array_equal(selected,reserve(output,base,config,population))
            with (output/'holdout_users.csv').open('a') as stream: stream.write('\n')
            with self.assertRaises(ValueError): reserve(output,base,config,population)

    def test_final_uses_validation_choice_and_cannot_be_repeated(self):
        from catboost import CatBoostRanker
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);base=root/'base';output=root/'strong'
            base.mkdir();output.mkdir()
            pd.Series([1,2],name='user_id').to_csv(base/'sample_users.csv',index=False)
            (base/'manifest.json').write_text(json.dumps({'config':{
                'popular_candidates':10,'genre_candidates':6,'windows':[7,30,90]}}))
            model=CatBoostRanker(iterations=2,depth=2,thread_count=1,
                                allow_writing_files=False,verbose=False)
            model.fit(pd.DataFrame({'item_count_30d':[0,1,0,1]}),[0,1,0,1],group_id=[1,1,2,2])
            model.save_model(str(base/'model.cbm'))
            for name,value in [('fixed',.2),('expanded',.3),('fixed/no_co',.1)]:
                folder=output/name;folder.mkdir(parents=True,exist_ok=True)
                model.save_model(str(folder/'model.cbm'))
                (folder/'metrics.json').write_text(json.dumps({'validation':{'catboost':{'ndcg@10':value}}}))
            config={**self.config,'holdout_users':2,'holdout_seed':42,
                    'feature_batch_users':1,'seed':42}
            events=pd.DataFrame({'user_id':[1,2,2,3,4,3,4],
                'item_id':[10,20,30,10,10,20,30],
                'date':pd.to_datetime(['2021-07-31']*5+['2021-08-01','2021-08-07'])})
            final(output,base,config,events,self.users,self.items)
            selection=json.loads((output/'selection.json').read_text())
            result=json.loads((output/'final/metrics.json').read_text())
            self.assertEqual(selection['primary_variant'],'expanded')
            self.assertEqual(result['users'],2)
            self.assertEqual(result['primary_variant'],'expanded')
            self.assertIn('primary_vs_original_pipeline',result)
            with self.assertRaises(ValueError): final(output,base,config,events,self.users,self.items)


if __name__=='__main__': unittest.main()
