import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from generative_docker import app
from generative_docker.io import ITEM_COLUMNS, USER_COLUMNS, read_inputs
from research.generative_recsys.src.data import synthetic_data


def fixture(directory):
    events,items,chosen=synthetic_data()
    events.to_csv(directory/'train.csv',index=False)
    test=pd.DataFrame([(int(u),100+(int(u)%4)*8+5,'2021-08-17') for u in chosen]
                      +[(1001,100,'2021-08-17'),(900,100,'2021-08-17')],columns=['user_id','item_id','date'])
    test.to_csv(directory/'test.csv',index=False)
    users=pd.DataFrame({'user_id':[*chosen,900,1001]})
    for key in USER_COLUMNS:
        if key not in users:users[key]='unknown'
    users.to_csv(directory/'users.csv',index=False)
    for key in ITEM_COLUMNS:
        if key not in items:items[key]='unknown'
    items.to_csv(directory/'items.csv',index=False)
    (directory/'dataset.json').write_text(json.dumps({'test_cutoff':'2021-08-16','test_end_exclusive':'2021-08-23'}))
    config={'sample_users':100,'embedding_dim':8,'rqvae_hidden':[8],'rqvae_latent_dim':4,
            'code_width':4,'rqvae_epochs':2,'rqvae_batch_size':16,'d_model':16,'heads':2,
            'layers':1,'batch_size':32,'epochs':1,'beam_width':32,'threads':2,'bootstrap_iterations':20}
    path=directory/'config.json';path.write_text(json.dumps(config))
    return path


class ContainerContractTests(unittest.TestCase):
    def test_train_then_load_without_training_and_corruption_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary);config=fixture(directory)
            run=directory/'first';run.mkdir()
            args=argparse.Namespace(config=config,device='cpu',inference_device='cpu')
            report=app.execute(args,directory,run)
            self.assertEqual(report['model_action'],'trained')
            self.assertEqual(report['overall']['users'],25)
            self.assertEqual(report['warm']['users'],24)
            self.assertEqual(report['cold']['users'],1)
            self.assertEqual(report['seen_test_pairs_excluded'],1)
            self.assertTrue((directory/'sid_model.pt').exists())
            first=pd.read_csv(run/'recommendations.csv')
            second=directory/'second';second.mkdir()
            args=argparse.Namespace(config=None,device='gpu',inference_device='cpu')
            with patch.object(app,'fit_bundle',side_effect=AssertionError('Must not retrain')):
                other=app.execute(args,directory,second)
            self.assertEqual(other['model_action'],'loaded')
            pd.testing.assert_frame_equal(first,pd.read_csv(second/'recommendations.csv'))
            self.assertEqual(report['overall'],other['overall'])
            metadata_path=directory/'sid_model.json'
            metadata=json.loads(metadata_path.read_text())
            future=dict(metadata,trained_before='2021-08-17')
            metadata_path.write_text(json.dumps(future))
            period=directory/'period';period.mkdir()
            with self.assertRaisesRegex(ValueError,'overlaps'):
                app.execute(args,directory,period)
            metadata_path.write_text(json.dumps(metadata))
            with (directory/'sid_model.pt').open('ab') as stream:stream.write(b'corruption')
            third=directory/'third';third.mkdir()
            with self.assertRaisesRegex(ValueError,'checksum'):
                app.execute(args,directory,third)

    def test_future_train_and_duplicate_pairs_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary);path=fixture(directory);config=app.load_config(path)
            events=pd.read_csv(directory/'train.csv');events.loc[0,'date']='2021-08-16'
            events.to_csv(directory/'train.csv',index=False)
            with self.assertRaisesRegex(ValueError,'train dates'):
                read_inputs(directory,config)
            events.loc[0,'date']='2021-06-01'
            pd.concat([events,events.iloc[:1]]).to_csv(directory/'train.csv',index=False)
            with self.assertRaisesRegex(ValueError,'duplicate'):
                read_inputs(directory,config)

    def test_unknown_config_and_wrong_model_period_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'bad.json';path.write_text('{"unknown":1}')
            with self.assertRaisesRegex(ValueError,'Unknown'):app.load_config(path)

    def test_context_excludes_models_data_and_unrelated_baseline_code(self):
        from generative_docker.prepare_context import stage
        with tempfile.TemporaryDirectory() as temporary:
            context=stage(Path(temporary)/'context')
            files=[str(p.relative_to(context)) for p in context.rglob('*') if p.is_file()]
            self.assertFalse(any('artifacts' in f or '.venv' in f or 'KION_DATASET' in f for f in files))
            self.assertNotIn('baseline/aggregates.py',files)
            self.assertIn('research/generative_recsys/src/rqvae.py',files)

    def test_export_exact_winning_weights_without_training(self):
        import torch
        from generative_docker.export_best import export_best, BEST
        if not (BEST/'model.pt').exists():self.skipTest('Local research weights not available')
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            export_best(directory)
            bundle=torch.load(directory/'sid_model.pt',map_location='cpu',weights_only=True)
            original=torch.load(BEST/'model.pt',map_location='cpu',weights_only=True)
            self.assertEqual(bundle['checkpoint']['epoch'],original['epoch'])
            for name,value in original['state_dict'].items():
                torch.testing.assert_close(value,bundle['checkpoint']['state_dict'][name],rtol=0,atol=0)
            self.assertEqual(len(bundle['item_ids']),14947)
            with self.assertRaisesRegex(ValueError,'overwritten'):export_best(directory)
