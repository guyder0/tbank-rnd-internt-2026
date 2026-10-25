import importlib.util
import unittest

import numpy as np
import pandas as pd

from research.generative_recsys.src.data import cap_history, prepare_queries, synthetic_data
from research.generative_recsys.src.decoding import SIDTrie, beam_search, exact_search
from research.generative_recsys.src.tokenizer import fit_codes, item_texts, lexical_embeddings


def configuration():
    return {'seed':42, 'code_width':4, 'code_levels':3, 'kmeans_iterations':5,
            'd_model':16, 'heads':2, 'layers':1, 'dropout':0., 'max_history':8}


class SIDTests(unittest.TestCase):
    def test_unique_deterministic_codes_with_identical_embeddings(self):
        ids = np.arange(7)
        embeddings = np.ones((len(ids),4), dtype='float32')
        first = fit_codes(ids, embeddings, configuration())
        second = fit_codes(ids, embeddings, configuration())
        np.testing.assert_array_equal(first['codes'],second['codes'])
        self.assertEqual(len({tuple(c) for c in first['codes']}),len(ids))
        self.assertEqual(first['diagnostics']['max_collision_bucket'],7)
        self.assertTrue((first['codes']>=3).all())
        self.assertLess(first['codes'].max(),first['vocab_size'])

    def test_random_control_preserves_prefix_tree_but_changes_assignment(self):
        embeddings=np.random.default_rng(42).normal(size=(20,8)).astype('float32')
        first=fit_codes(np.arange(20),embeddings,configuration())
        other=fit_codes(np.arange(20),embeddings,{**configuration(),'random_codes':True})
        self.assertEqual(sorted(map(tuple,first['codes'])),sorted(map(tuple,other['codes'])))
        self.assertFalse(np.array_equal(first['codes'],other['codes']))

    def test_lexical_embeddings_finite_and_deterministic(self):
        _,items,_=synthetic_data()
        texts=item_texts(items)
        first=lexical_embeddings(texts,8,42)
        second=lexical_embeddings(texts,8,42)
        np.testing.assert_allclose(first,second,atol=1e-5)
        self.assertEqual(first.shape,(32,8))
        self.assertTrue(np.isfinite(first).all())

    def test_beam_uses_raw_scores_and_excludes_seen_paths(self):
        trie=SIDTrie([10,20,30],np.array([[3,6],[4,7],[4,8]]))
        def callback(prefixes):
            result=np.full((len(prefixes),10),-100.)
            for n,prefix in enumerate(prefixes):
                if not prefix: result[n,3],result[n,4]=-.4,-.5
                elif prefix==(3,): result[n,6]=-4.
                else: result[n,7],result[n,8]=-.2,-.3
            return result
        result=beam_search(callback,trie,3,2)
        self.assertEqual([r[0] for r in result],[20,30])
        self.assertAlmostEqual(result[0][1],-.7)
        self.assertEqual(result,exact_search(callback,trie,2))
        self.assertEqual([r[0] for r in beam_search(callback,trie,3,2,{20,30})],[10])
        self.assertEqual(beam_search(callback,trie,3,2,{10,20,30}),[])

    def test_empty_recommendations_and_retrieval_misses_keep_denominator(self):
        from research.generative_recsys.src.metrics import evaluate_lists
        queries=[{'user_id':1,'positives':[10,20]}, {'user_id':2,'positives':[30]}]
        metrics,individual=evaluate_lists(queries,[[10],[]],10)
        self.assertEqual(metrics['users'],2)
        self.assertEqual(metrics['recall@10'],.25)
        self.assertEqual(len(individual),2)

    def test_duplicate_sid_rejected(self):
        with self.assertRaises(ValueError): SIDTrie([1,2],np.array([[3,4],[3,4]]))

    def test_complete_day_cap_is_order_invariant(self):
        q={'items':np.array([10,20,30,40]),'ages':np.array([9,2,2,1])}
        ids,ages=cap_history(q,2)
        self.assertEqual(set(ids),{20,30,40})
        self.assertEqual(sorted(ages),[1,2,2])

    def test_past_inputs_and_catalog_invariant_to_future_outcomes(self):
        events,items,chosen=synthetic_data()
        dates=['2021-07-19','2021-07-26','2021-08-02','2021-08-09']
        catalog,splits=prepare_queries(events,items,chosen,dates,10)
        changed=events.copy()
        future=changed.date>=pd.Timestamp(dates[-1])
        changed.loc[future,'item_id']=np.roll(changed.loc[future,'item_id'],1)
        other_catalog,other=prepare_queries(changed,items,chosen,dates,10)
        pd.testing.assert_frame_equal(catalog,other_catalog)
        for first,second in zip(splits[-1]['queries'],other[-1]['queries']):
            self.assertEqual(first['user_id'],second['user_id'])
            np.testing.assert_array_equal(first['items'],second['items'])
            np.testing.assert_array_equal(first['ages'],second['ages'])
            self.assertEqual(first['popularity'],second['popularity'])
            self.assertTrue((first['ages']>0).all())


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch is not installed')
class TorchTests(unittest.TestCase):
    def setUp(self):
        import torch
        from research.generative_recsys.src.model import SIDGenerator
        torch.manual_seed(42); torch.set_num_threads(2)
        self.torch=torch
        self.model=SIDGenerator(24,4,configuration()).eval()
        self.history=torch.tensor([[[3,7,11,15],[4,8,12,16]]])
        self.ages=torch.tensor([[2.,2.]])
        self.padding=torch.tensor([[False,False]])
        self.target=torch.tensor([[5,9,13,17]])

    def test_cuda_request_fails_explicitly_when_unavailable(self):
        from unittest.mock import patch
        from research.generative_recsys.run import preflight
        with patch('torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'No CPU fallback'):
                preflight('cuda')

    def test_forward_backward_finite(self):
        output=self.model(self.history,self.ages,self.padding,self.target)
        self.assertEqual(tuple(output.shape),(1,4,24))
        loss=self.torch.nn.functional.cross_entropy(output.transpose(1,2),self.target)
        loss.backward()
        self.assertTrue(self.torch.isfinite(loss))
        self.assertTrue(all(self.torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None))

    def test_same_day_permutation_and_padding_do_not_change_predictions(self):
        torch=self.torch
        with torch.inference_mode():
            expected=self.model(self.history,self.ages,self.padding,self.target)
            permuted=self.model(self.history.flip(1),self.ages.flip(1),self.padding,self.target)
            torch.testing.assert_close(expected,permuted,atol=1e-5,rtol=1e-5)
            history=torch.cat([self.history,torch.zeros((1,1,4),dtype=torch.long)],dim=1)
            ages=torch.cat([self.ages,torch.zeros((1,1))],dim=1)
            padding=torch.tensor([[False,False,True]])
            torch.testing.assert_close(expected,self.model(history,ages,padding,self.target),atol=1e-5,rtol=1e-5)

    def test_decoder_cannot_see_future_target_tokens(self):
        with self.torch.inference_mode():
            before=self.model(self.history,self.ages,self.padding,self.target)
            changed=self.target.clone();changed[:,2]=18
            after=self.model(self.history,self.ages,self.padding,changed)
        self.torch.testing.assert_close(before[:,:3],after[:,:3],atol=1e-6,rtol=1e-6)

    def test_tiny_overfit(self):
        torch=self.torch
        optimizer=torch.optim.Adam(self.model.parameters(),lr=.02)
        initial=None
        for _ in range(60):
            optimizer.zero_grad()
            output=self.model(self.history,self.ages,self.padding,self.target)
            loss=torch.nn.functional.cross_entropy(output.transpose(1,2),self.target)
            if initial is None: initial=float(loss.detach())
            loss.backward();optimizer.step()
        self.assertLess(float(loss.detach()),initial*.1)

    def test_empty_recommendations_keep_users_in_denominator(self):
        from research.generative_recsys.src.metrics import evaluate_lists
        metrics,individual=evaluate_lists([{'user_id':1,'positives':[10]}],[[]],10)
        self.assertEqual(metrics['users'],1)
        self.assertEqual(metrics['ndcg@10'],0)


if __name__=='__main__': unittest.main()
