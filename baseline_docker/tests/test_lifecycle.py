import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile
from unittest.mock import patch

import numpy as np
import pandas as pd

from baseline.aggregates import ManualAggregates
from baseline.pipeline import make_candidates, make_features
from baseline.refined import RefinedFeatures
from baseline_docker import app
from baseline_docker.features import CandidateFactory, FeatureBuilder
from baseline_docker.prepare_context import stage
from baseline_docker.prepare_kion import prepare
from baseline_docker.export_best import BEST, export_best


def fixture(directory):
    users = pd.DataFrame({'user_id':[1,2,3,4],'age':['a']*4,'income':['i']*4,
                          'sex':['M','F','M','F'],'kids_flg':[0,1,0,0]})
    items = pd.DataFrame({'item_id':[10,20,30,40,50,60,70],'content_type':['film']*7,
                          'release_year':[2020]*7,'age_rating':[16]*7,'for_kids':[0]*7,
                          'genres':['a','a, b','b','a','b','c','c'],
                          'countries':['c']*7,'directors':['d']*7,'actors':['x']*7,'studios':['s']*7})
    rows = [(1,10,'2021-01-01'),(2,20,'2021-01-01'),(3,30,'2021-01-01'),
            (4,40,'2021-01-01'),(4,50,'2021-01-02'),
            (1,20,'2021-02-03'),(2,30,'2021-02-03'),(3,10,'2021-02-03'),
            (1,30,'2021-02-10'),(2,10,'2021-02-10'),(3,20,'2021-02-10')]
    train = pd.DataFrame(rows,columns=['user_id','item_id','last_watch_dt'])
    test = pd.DataFrame([(1,40,'2021-02-16'),(2,50,'2021-02-17'),(3,60,'2021-02-16'),
                         (3,10,'2021-02-16'),(9,10,'2021-02-16')],columns=train.columns)
    for name, frame in [('train',train),('test',test),('users',users),('items',items)]:
        frame.to_csv(directory/f'{name}.csv',index=False)
    (directory/'dataset.json').write_text(json.dumps({'test_cutoff':'2021-02-16','test_end_exclusive':'2021-02-23'}))
    (directory/'config.json').write_text(json.dumps({'iterations':3,'depth':2,'thread_count':1,
        'train_weeks':1,'sample_users':4,'minimum_support':1,'neighbors':3,
        'feature_batch_users':2,'bootstrap_iterations':20}))


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        fixture(self.root)
        self.config = app.load_config(self.root/'config.json')

    def tearDown(self):
        self.temporary.cleanup()

    def execute(self, number, config=True):
        from argparse import Namespace
        run = self.root/f'run{number}'; run.mkdir()
        args = Namespace(config=self.root/'config.json' if config else None, device='cpu', devices='0')
        with contextlib.redirect_stdout(io.StringIO()):
            return app.execute(args,self.root,run)

    def test_train_commit_then_load_without_training_or_gpu(self):
        first = self.execute(1)
        self.assertEqual(first['model_action'],'trained')
        self.assertEqual(first['training_backend'],'CPU')
        self.assertEqual(first['overall']['users'],4)
        self.assertEqual(first['warm']['users'],3)
        self.assertEqual(first['cold']['users'],1)
        self.assertEqual(first['seen_test_pairs_excluded'],1)
        self.assertEqual(first['overall']['complete_retrieval_miss_users'],1)
        self.assertTrue((self.root/'model.json').exists())
        before = app.sha256(self.root/'model.cbm')
        with patch.object(app,'fit_model',side_effect=AssertionError('Must not retrain')), \
             patch.object(app,'choose_backend',side_effect=AssertionError('Must not probe GPU')):
            second = self.execute(2,config=False)
        self.assertEqual(second['model_action'],'loaded')
        self.assertEqual(before,app.sha256(self.root/'model.cbm'))
        self.assertEqual(first['overall'],second['overall'])
        recommendations = pd.read_csv(self.root/'run2/recommendations.csv')
        past = pd.read_csv(self.root/'train.csv')
        self.assertTrue(recommendations.merge(past,on=['user_id','item_id']).empty)

    def test_input_errors_are_clear(self):
        test = pd.read_csv(self.root/'test.csv')
        test.loc[0,'last_watch_dt']='2021-01-01';test.to_csv(self.root/'test.csv',index=False)
        with self.assertRaisesRegex(ValueError,'Require train dates'):
            app.read_inputs(self.root,self.config)
        fixture(self.root)
        train = pd.read_csv(self.root/'train.csv')
        pd.concat([train,train.iloc[:1]]).to_csv(self.root/'train.csv',index=False)
        with self.assertRaisesRegex(ValueError,'duplicate pairs'):
            app.read_inputs(self.root,self.config)
        (self.root/'test.csv').unlink()
        with self.assertRaisesRegex(ValueError,'Missing required input files.*test.csv'):
            app.read_inputs(self.root,self.config)

    def test_auto_cpu_fallback_and_strict_gpu(self):
        failure = subprocess.CompletedProcess([],1,stdout='',stderr='CUDA unavailable')
        with patch.object(app.subprocess,'run',return_value=failure):
            backend, reason = app.choose_backend('auto','0',self.config)
            self.assertEqual(backend,'CPU');self.assertIn('CUDA unavailable',reason)
            with self.assertRaisesRegex(RuntimeError,'explicitly requested'):
                app.choose_backend('gpu','0',self.config)
        with patch.object(app.subprocess,'run',side_effect=AssertionError('No probe on explicit CPU')):
            self.assertEqual(app.choose_backend('cpu','0',self.config),('CPU',None))

    def test_model_is_committed_even_when_test_evaluation_fails(self):
        test = pd.DataFrame({'user_id':[1],'item_id':[10],'last_watch_dt':['2021-02-16']})
        test.to_csv(self.root/'test.csv',index=False)
        with self.assertRaisesRegex(ValueError,'no new user-item pairs'):
            self.execute(1)
        self.assertTrue((self.root/'model.cbm').exists())
        self.assertTrue((self.root/'model.json').exists())

    def test_cached_model_rejects_checksum_and_temporal_leakage(self):
        self.execute(1)
        metadata = json.loads((self.root/'model.json').read_text())
        metadata['trained_before']='2021-03-01'
        (self.root/'model.json').write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError,'later than this test cutoff'):
            self.execute(2,config=False)
        metadata['trained_before']='2021-02-16';metadata['model_sha256']='bad'
        (self.root/'model.json').write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError,'checksum'):
            self.execute(3,config=False)

    def test_bare_saved_model_can_be_evaluated(self):
        self.execute(1)
        (self.root/'model.json').unlink()
        with patch.object(app,'fit_model',side_effect=AssertionError('No retraining')):
            result = self.execute(2)
        self.assertFalse(result['model_provenance_verified'])
        self.assertEqual(result['model_action'],'loaded')

    def test_feature_and_candidate_parity_with_baseline(self):
        events, _, users, items, _, _, _ = app.read_inputs(self.root,self.config)
        cutoff = pd.Timestamp('2021-02-02')
        history = events.loc[events.date<cutoff]
        ids = np.array([1,2,3])
        expected_original = make_candidates(history,ids,items,self.config,cutoff)
        actual_original = CandidateFactory(history,ids,items,cutoff,self.config).build(ids)
        pd.testing.assert_frame_equal(expected_original,actual_original,check_dtype=False)
        ctx = ManualAggregates(history,users,items,cutoff,self.config,user_ids=ids)
        candidates = ctx.expand_candidates(expected_original)
        expected = pd.concat([make_features(candidates,history,users,items,cutoff,self.config['windows']),
                              ctx.features(candidates),RefinedFeatures(ctx,expected_original).features(candidates)],axis=1)
        expected = expected[['user_id','item_id',*self.config['feature_columns']]]
        builder = FeatureBuilder(history,users,items,cutoff,ids,self.config)
        actual = pd.concat([builder.build(ids[:2]),builder.build(ids[2:])],ignore_index=True)
        pd.testing.assert_frame_equal(expected,actual,check_dtype=False)

    def test_staged_runtime_has_no_external_artifact_dependencies(self):
        directory = stage(self.root/'context')
        expected_baseline={'__init__.py','pipeline.py','aggregates.py','refined.py'}
        self.assertEqual({p.name for p in (directory/'baseline').iterdir()},expected_baseline)
        self.assertFalse((directory/'KION_DATASET').exists())
        process = subprocess.run([sys.executable,'-m','baseline_docker.app','--data-dir',str(self.root),
                                  '--device','auto','--config',str(self.root/'config.json')],
                                  cwd=directory,capture_output=True,text=True,timeout=90)
        self.assertEqual(process.returncode,0,process.stdout+'\n'+process.stderr)
        status_files=list((self.root/'runs').glob('*/status.json'))
        self.assertEqual(len(status_files),1)
        self.assertEqual(json.loads(status_files[0].read_text())['status'],'completed')
        self.assertIn('Detailed test metrics',(status_files[0].parent/'run.log').read_text())

    def test_kion_preparation_preserves_chronology_and_refuses_overwrite(self):
        archive = self.root/'input.zip'
        with zipfile.ZipFile(archive,'w') as zipped:
            train = pd.read_csv(self.root/'train.csv')
            test = pd.read_csv(self.root/'test.csv')
            # Real KION aggregate has one row per pair over the entire archive.
            interactions = pd.concat([train,test]).drop_duplicates(['user_id','item_id'])
            zipped.writestr('data_original/interactions.csv',interactions.to_csv(index=False))
            for name in ['users.csv','items.csv']:
                zipped.write(self.root/name,'data_original/'+name)
        output = self.root/'prepared'
        counts = prepare(archive,output,'2021-02-16')
        self.assertGreater(counts['train'],0);self.assertGreater(counts['test'],0)
        app.read_inputs(output,self.config)
        with self.assertRaisesRegex(ValueError,'already exists'):
            prepare(archive,output,'2021-02-16')

    @unittest.skipUnless((BEST/'model.cbm').exists(),'Original GPU artifact unavailable')
    def test_original_best_gpu_model_predicts_on_cpu_without_retraining(self):
        # Move the fixture after the original model's validation-selection period.
        for name in ['train.csv','test.csv']:
            frame = pd.read_csv(self.root/name)
            frame.last_watch_dt = pd.to_datetime(frame.last_watch_dt)+pd.DateOffset(years=1)
            frame.to_csv(self.root/name,index=False)
        (self.root/'dataset.json').write_text(json.dumps({'test_cutoff':'2022-02-16','test_end_exclusive':'2022-02-23'}))
        export_best(self.root)
        with patch.object(app,'choose_backend',side_effect=AssertionError('No GPU initialization')), \
             patch.object(app,'fit_model',side_effect=AssertionError('No retraining')):
            result = self.execute(1,config=False)
        self.assertEqual(result['model_action'],'loaded')
        self.assertEqual(result['training_backend'],'GPU')
        self.assertEqual(result['inference_backend'],'CPU')

    def test_directory_lock_prevents_concurrent_writers(self):
        with app.directory_lock(self.root):
            with self.assertRaisesRegex(RuntimeError,'Another container'):
                with app.directory_lock(self.root):
                    pass

    def test_no_candidates_still_count_as_zero_in_test_metrics(self):
        from unittest.mock import Mock
        train, test, users, items, cutoff, _, _ = app.read_inputs(self.root,self.config)
        builder = Mock()
        builder.build.return_value = pd.DataFrame(columns=['user_id','item_id',*self.config['feature_columns']])
        model = Mock()
        model.predict.side_effect = AssertionError('No prediction for empty candidate batch')
        with patch.object(app,'FeatureBuilder',return_value=builder), contextlib.redirect_stdout(io.StringIO()):
            report = app.evaluate_test(model,train,test,users,items,cutoff,self.config,self.root)
        self.assertEqual(report['overall']['users'],4)
        self.assertEqual(report['overall']['no_candidate_users'],4)
        self.assertEqual(report['overall']['metrics']['ndcg@10'],0)
        self.assertEqual(report['catalogue_coverage@10'],0)

    def test_changing_test_labels_does_not_change_training_or_tree_selection(self):
        self.execute(1)
        expected = json.loads((self.root/'run1/training.json').read_text())
        first = app.CatBoostRanker().load_model(str(self.root/'model.cbm'))
        other = self.root/'other';other.mkdir();fixture(other)
        test = pd.read_csv(other/'test.csv')
        test.loc[0,'item_id']=50
        test.to_csv(other/'test.csv',index=False)
        run = other/'run';run.mkdir()
        from argparse import Namespace
        args = Namespace(config=other/'config.json',device='cpu',devices='0')
        with contextlib.redirect_stdout(io.StringIO()):
            app.execute(args,other,run)
        second = app.CatBoostRanker().load_model(str(other/'model.cbm'))
        self.assertEqual(expected,json.loads((run/'training.json').read_text()))
        np.testing.assert_array_equal(first.get_leaf_values(),second.get_leaf_values())


if __name__=='__main__':
    unittest.main()
