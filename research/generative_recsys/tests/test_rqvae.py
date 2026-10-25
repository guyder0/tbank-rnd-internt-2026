import importlib.util
import tempfile
import unittest

import numpy as np

from research.generative_recsys.src.tokenizer import fit_codes, validate_prepared_config


class PreparedCompatibilityTests(unittest.TestCase):
    def test_old_cache_not_accepted_as_rqvae(self):
        with self.assertRaisesRegex(ValueError, 'old k-means'):
            validate_prepared_config({'tokenizer':'rqvae'}, {})
        validate_prepared_config({'tokenizer':'residual_kmeans'}, {})

    def test_changed_rqvae_settings_rejected(self):
        config={'tokenizer':'rqvae','rqvae_beta':.25}
        with self.assertRaises(ValueError):
            validate_prepared_config(config, {**config,'rqvae_beta':.5})


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch not installed')
class RQVAETests(unittest.TestCase):
    def setUp(self):
        import torch
        from research.generative_recsys.src.rqvae import RQVAE
        self.torch=torch
        torch.manual_seed(42);torch.set_num_threads(2)
        self.model=RQVAE(2, [], 2, 2, 2, beta=.25)
        with torch.no_grad():
            self.model.codebooks.copy_(torch.tensor([[[1.,0.],[0.,1.]],[[.25,0.],[0.,.25]]]))

    def test_residual_quantization_and_loss_match_hand_calculation(self):
        z=self.torch.tensor([[1.3,.1]],requires_grad=True)
        quantized,codes,book,commitment=self.model.quantize(z)
        self.torch.testing.assert_close(quantized,self.torch.tensor([[1.25,0.]]))
        self.assertEqual(codes.tolist(),[[0,0]])
        self.assertAlmostEqual(float(book.detach()),.1125,places=6)
        self.assertAlmostEqual(float(commitment.detach()),.25*.1125,places=6)

    def test_straight_through_gradient_and_stop_gradients(self):
        torch=self.torch
        z=torch.tensor([[1.3,.1]],requires_grad=True)
        quantized,_,_,_=self.model.quantize(z)
        quantized.sum().backward()
        torch.testing.assert_close(z.grad,torch.ones_like(z))
        self.assertIsNone(self.model.codebooks.grad)
        z=torch.tensor([[1.3,.1]],requires_grad=True)
        _,_,book,_=self.model.quantize(z)
        book.backward()
        self.assertIsNone(z.grad)
        self.assertGreater(float(self.model.codebooks.grad.abs().sum()),0)
        self.model.zero_grad(set_to_none=True)
        z=torch.tensor([[1.3,.1]],requires_grad=True)
        _,_,_,commitment=self.model.quantize(z)
        commitment.backward()
        self.assertGreater(float(z.grad.abs().sum()),0)
        self.assertIsNone(self.model.codebooks.grad)

    def test_reconstruction_loss_is_squared_l2_and_all_components_train(self):
        x=self.torch.tensor([[.1,.2],[.3,.4]])
        reconstructed,_,losses=self.model(x)
        expected=(reconstructed-x).square().sum(1).mean()
        self.torch.testing.assert_close(losses['reconstruction'],expected)
        losses['total'].backward()
        for module in [self.model.encoder,self.model.decoder]:
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters()))
        self.assertGreater(float(self.model.codebooks.grad.abs().sum()),0)
        self.assertTrue(all(self.torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None))

    def test_tiny_reconstruction_improves(self):
        torch=self.torch
        from research.generative_recsys.src.rqvae import RQVAE
        model=RQVAE(2,[8],2,2,4)
        x=torch.tensor([[1.,0.],[.9,.1],[0.,1.],[.1,.9]])
        model.initialize_codebooks(x,42,5)
        initial=float(model(x)[2]['reconstruction'].detach())
        optimizer=torch.optim.Adam(model.parameters(),lr=.01)
        for _ in range(80):
            optimizer.zero_grad();losses=model(x)[2]
            losses['total'].backward();optimizer.step()
        self.assertLess(float(model(x)[2]['reconstruction'].detach()),initial*.5)

    def test_full_fit_suffix_and_checkpoint(self):
        config={'seed':42,'tokenizer':'rqvae','threads':2,'code_width':4,'code_levels':3,
                'kmeans_iterations':5,'rqvae_hidden':[8],'rqvae_latent_dim':4,
                'rqvae_beta':.25,'rqvae_batch_size':16,'rqvae_epochs':3,
                'rqvae_optimizer':'adagrad','rqvae_learning_rate':.05,'rqvae_max_steps':4}
        x=np.random.default_rng(42).normal(size=(12,6)).astype('float32')
        with tempfile.TemporaryDirectory() as output:
            codes=fit_codes(np.arange(len(x)),x,config,output=output,log=lambda value:None)
            self.assertEqual(codes['codes'].shape,(12,4))
            self.assertEqual(len(set(map(tuple,codes['codes']))),12)
            self.assertEqual(codes['tokenizer'],'rqvae')
            self.assertTrue(codes['diagnostics']['checkpoint_reload'])


if __name__=='__main__': unittest.main()


class DiagnosticsTests(unittest.TestCase):
    def test_constant_mean_decoder_has_zero_gain_and_variance(self):
        import torch
        from research.generative_recsys.src.rqvae import RQVAE, reconstruction_diagnostics
        model = RQVAE(2, [], 2, 2, 2)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.decoder[0].bias.fill_(.5)
        report = reconstruction_diagnostics(model,np.eye(2,dtype='float32'),1,'cpu')
        self.assertAlmostEqual(report['mean_baseline_mse'],.5)
        self.assertAlmostEqual(report['catalog_losses']['reconstruction'],.5)
        self.assertAlmostEqual(report['reconstruction_gain_vs_mean'],0.)
        self.assertAlmostEqual(report['reconstruction_variance'],0.)
        self.assertEqual([level['used'] for level in report['usage']],[1,1])
        self.assertEqual(report['usage'][0]['counts'],[2,0])


class AlternatingOptimizationTests(unittest.TestCase):
    def setUp(self):
        import torch
        from research.generative_recsys.src.rqvae import RQVAE
        self.torch = torch
        torch.manual_seed(42)
        self.model = RQVAE(3,[8],2,2,4)
        self.batch = torch.randn(7,3)

    def test_codebook_step_does_not_update_network(self):
        from research.generative_recsys.src.rqvae import update_codebooks
        torch = self.torch
        before = {name:p.detach().clone() for name,p in self.model.named_parameters()}
        optimizer = torch.optim.Adagrad([self.model.codebooks],lr=.1,initial_accumulator_value=.1)
        update_codebooks(self.model,self.batch,optimizer)
        self.assertFalse(torch.equal(self.model.codebooks,before['codebooks']))
        for name,p in self.model.named_parameters():
            if name!='codebooks':
                torch.testing.assert_close(p,before[name],rtol=0,atol=0)
                self.assertIsNone(p.grad)

    def test_network_step_matches_reconstruction_only_and_freezes_books(self):
        import copy
        from research.generative_recsys.src.rqvae import update_network
        torch = self.torch
        expected = copy.deepcopy(self.model)
        parameters = lambda m:[*m.encoder.parameters(),*m.decoder.parameters()]
        actual_opt = torch.optim.Adagrad(parameters(self.model),lr=.1,initial_accumulator_value=.1)
        expected_opt = torch.optim.Adagrad(parameters(expected),lr=.1,initial_accumulator_value=.1)
        _,_,losses = expected(self.batch)
        losses['reconstruction'].backward();expected_opt.step()
        books = self.model.codebooks.detach().clone()
        update_network(self.model,self.batch,actual_opt)
        for actual,target in zip(self.model.parameters(),expected.parameters()):
            torch.testing.assert_close(actual,target,rtol=0,atol=0)
        torch.testing.assert_close(self.model.codebooks,books,rtol=0,atol=0)
        self.assertTrue(self.model.codebooks.requires_grad)
        self.assertIsNone(self.model.codebooks.grad)


class AffineCodebookTests(unittest.TestCase):
    def test_identity_affine_matches_existing_quantizer(self):
        import torch
        from research.generative_recsys.src.rqvae import RQVAE
        torch.manual_seed(7)
        old = RQVAE(2,[],2,2,4)
        torch.manual_seed(7)
        affine = RQVAE(2,[],2,2,4,affine=True)
        z = torch.randn(5,2)
        for expected,actual in zip(old.quantize(z),affine.quantize(z)):
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    def test_unselected_effective_centers_move_through_shared_affine(self):
        import torch
        from research.generative_recsys.src.rqvae import RQVAE
        model = RQVAE(2,[],2,1,3,affine=True)
        with torch.no_grad():
            model.codebooks.copy_(torch.tensor([[[0.,0.],[10.,10.],[20.,20.]]]))
        before = model.effective_codebooks().detach().clone()
        raw = model.codebooks.detach().clone()
        _,codes,loss,_ = model.quantize(torch.tensor([[1.,2.]]))
        self.assertEqual(codes.tolist(),[[0]])
        optimizer = torch.optim.SGD(model.codebook_parameters(),lr=.1)
        loss.backward()
        torch.testing.assert_close(model.codebooks.grad[0,1:],torch.zeros(2,2))
        self.assertGreater(float(model.affine_shift.grad.abs().sum()),0.)
        optimizer.step()
        torch.testing.assert_close(model.codebooks[0,1:],raw[0,1:],rtol=0,atol=0)
        self.assertFalse(torch.equal(model.effective_codebooks()[0,1:],before[0,1:]))

    def test_network_step_freezes_base_and_affine_parameters_and_reload(self):
        import copy
        import torch
        from research.generative_recsys.src.rqvae import RQVAE, update_network
        model = RQVAE(3,[8],2,2,4,affine=True)
        before = [p.detach().clone() for p in model.codebook_parameters()]
        optimizer = torch.optim.Adagrad([*model.encoder.parameters(),*model.decoder.parameters()],lr=.1)
        update_network(model,torch.randn(5,3),optimizer)
        for actual,expected in zip(model.codebook_parameters(),before):
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            self.assertIsNone(actual.grad)
            self.assertTrue(actual.requires_grad)
        restored = RQVAE(3,[8],2,2,4,affine=True)
        restored.load_state_dict(copy.deepcopy(model.state_dict()))
        torch.testing.assert_close(model.effective_codebooks(),restored.effective_codebooks(),rtol=0,atol=0)
