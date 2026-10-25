"""Residual-quantized autoencoder following TIGER §3.1.

Squared L2 reconstruction + per-level codebook and beta commitment terms.
Reconstruction uses a straight-through estimator; residual assignments have
no per-level normalization. K-means only initializes the learned codebooks.
"""
import json
from pathlib import Path
import time
import warnings

import numpy as np
from scipy.cluster.vq import kmeans2
import torch
from torch import nn
from torch.nn import functional as F


def mlp(sizes):
    layers = []
    for number, (left, right) in enumerate(zip(sizes, sizes[1:])):
        layers.append(nn.Linear(left, right))
        if number < len(sizes)-2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class RQVAE(nn.Module):
    def __init__(self, input_dim, hidden, latent_dim, levels, width, beta=.25):
        super().__init__()
        self.encoder = mlp([input_dim, *hidden, latent_dim])
        self.decoder = mlp([latent_dim, *reversed(hidden), input_dim])
        self.codebooks = nn.Parameter(torch.randn(levels, width, latent_dim)*.01)
        self.beta = beta

    def quantize(self, z):
        residual, summed = z, torch.zeros_like(z)
        indices, codebook_loss, commitment_loss = [], z.new_zeros(()), z.new_zeros(())
        for book in self.codebooks:
            distances = residual.square().sum(1, keepdim=True)+book.square().sum(1)[None]-2*residual@book.T
            code = distances.argmin(1)
            selected = F.embedding(code, book)
            codebook_loss = codebook_loss+(residual.detach()-selected).square().sum(1).mean()
            commitment_loss = commitment_loss+self.beta*(residual-selected.detach()).square().sum(1).mean()
            indices.append(code)
            summed = summed+selected
            # Commitment updates encoder, not earlier codebooks; reconstruction
            # updates encoder/decoder through the straight-through path below.
            residual = residual-selected.detach()
        straight_through = z+(summed-z).detach()
        return straight_through, torch.stack(indices, dim=1), codebook_loss, commitment_loss

    def forward(self, x):
        quantized, codes, book, commitment = self.quantize(self.encoder(x))
        reconstruction = self.decoder(quantized)
        recon = (reconstruction-x).square().sum(1).mean()
        return reconstruction, codes, {'reconstruction': recon, 'codebook': book,
                                       'commitment': commitment, 'total': recon+book+commitment}

    @torch.no_grad()
    def initialize_codebooks(self, x, seed, iterations=25):
        residual = self.encoder(x).cpu().numpy().copy()
        for level in range(len(self.codebooks)):
            count = min(self.codebooks.shape[1], len(np.unique(residual, axis=0)))
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='One of the clusters is empty')
                centers, _ = kmeans2(residual, max(1,count), iter=iterations,
                                     minit='++', seed=seed+level)
            # Tiny fixtures can have fewer distinct vectors than codebook rows.
            centers = centers[np.arange(self.codebooks.shape[1]) % len(centers)]
            book = torch.as_tensor(centers, device=self.codebooks.device)
            self.codebooks[level].copy_(book)
            distance = ((residual[:,None]-centers[None])**2).sum(-1)
            residual = residual-centers[distance.argmin(1)]


@torch.no_grad()
def encode_all(model, embeddings, batch_size, device):
    model.eval()
    parts = []
    recon_sum = 0.
    for start in range(0, len(embeddings), batch_size):
        x = torch.as_tensor(embeddings[start:start+batch_size], device=device)
        _, codes, losses = model(x)
        if not torch.isfinite(losses['total']):
            raise ValueError('Non-finite RQ-VAE encoding loss')
        parts.append(codes.cpu().numpy())
        recon_sum += float(losses['reconstruction'])*len(x)
    return np.concatenate(parts), recon_sum/len(embeddings)


@torch.no_grad()
def reconstruction_diagnostics(model, embeddings, batch_size, device):
    """Full-catalog evaluation, separated from minibatch training losses."""
    model.eval()
    latent, reconstructed, assignments = [], [], []
    losses = {name:0. for name in ['reconstruction','codebook','commitment','total']}
    for start in range(0,len(embeddings),batch_size):
        x = torch.as_tensor(embeddings[start:start+batch_size],device=device)
        z = model.encoder(x)
        quantized, codes, book, commitment = model.quantize(z)
        output = model.decoder(quantized)
        reconstruction = (output-x).square().sum(1).mean()
        values = dict(reconstruction=reconstruction,codebook=book,commitment=commitment,
                      total=reconstruction+book+commitment)
        if not all(torch.isfinite(value) for value in values.values()):
            raise ValueError('Non-finite RQ-VAE diagnostics')
        for name,value in values.items():
            losses[name] += float(value)*len(x)/len(embeddings)
        latent.append(z.cpu().numpy()); reconstructed.append(output.cpu().numpy())
        assignments.append(codes.cpu().numpy())
    z = np.concatenate(latent).astype('float64')
    output = np.concatenate(reconstructed).astype('float64')
    x = np.asarray(embeddings,dtype='float64')
    codes = np.concatenate(assignments)
    variance = lambda values: float(np.var(values,axis=0).sum())
    baseline = variance(x)
    usage = []
    for level in range(codes.shape[1]):
        counts = np.bincount(codes[:,level],minlength=model.codebooks.shape[1])
        probabilities = counts[counts>0]/len(codes)
        usage.append({'used':int((counts>0).sum()),'counts':counts.tolist(),
            'effective_codes':float(np.exp(-(probabilities*np.log(probabilities)).sum())),
            'largest_code_fraction':float(counts.max()/len(codes))})
    return {'catalog_losses':losses,'mean_baseline_mse':baseline,
        'reconstruction_gain_vs_mean':1-losses['reconstruction']/baseline if baseline>0 else None,
        'input_variance':baseline,'latent_variance':variance(z),
        'reconstruction_variance':variance(output),
        'prediction_mean_bias_l2':float(np.linalg.norm(output.mean(0)-x.mean(0))),
        'usage':usage}


def update_codebooks(model, batch, optimizer):
    """Inner step: fit codebooks to fixed current encoder outputs."""
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        latent = model.encoder(batch)
    _, _, loss, _ = model.quantize(latent)
    if not torch.isfinite(loss):
        raise ValueError('Non-finite codebook loss')
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def update_network(model, batch, optimizer):
    """Outer step: reconstruction only, with fresh assignments and fixed books.

    Huh et al. §4.2 Eq.18 optimizes the task loss, not joint commitment.
    """
    model.zero_grad(set_to_none=True)
    previous = model.codebooks.requires_grad
    model.codebooks.requires_grad_(False)
    try:
        _, _, losses = model(batch)
        if not torch.isfinite(losses['total']):
            raise ValueError('Non-finite network loss')
        losses['reconstruction'].backward()
        optimizer.step()
    finally:
        model.codebooks.requires_grad_(previous)
    return losses


def train_rqvae(embeddings, config, output, device, log):
    started = time.monotonic()
    embeddings = np.asarray(embeddings, dtype='float32')
    if embeddings.ndim != 2 or not len(embeddings) or not np.isfinite(embeddings).all():
        raise ValueError('RQ-VAE requires nonempty finite embedding matrix')
    torch.manual_seed(config['seed'])
    torch.set_num_threads(config['threads'])
    settings = {'input_dim': embeddings.shape[1], 'hidden': config['rqvae_hidden'],
                'latent_dim': config['rqvae_latent_dim'], 'levels': config['code_levels'],
                'width': config['code_width'], 'beta': config['rqvae_beta']}
    model = RQVAE(**settings).to(device)
    batch_size = config['rqvae_batch_size']
    rng = np.random.default_rng(config['seed'])
    initial_batch = rng.permutation(len(embeddings))[:batch_size]
    model.initialize_codebooks(torch.as_tensor(embeddings[initial_batch], device=device),
                               config['seed'], config['kmeans_iterations'])
    initial_codes, initial_reconstruction = encode_all(model, embeddings, batch_size, device)
    def make_optimizer(parameters):
        if config['rqvae_optimizer'] == 'adagrad':
            return torch.optim.Adagrad(parameters, lr=config['rqvae_learning_rate'],initial_accumulator_value=.1)
        if config['rqvae_optimizer'] == 'adam':
            return torch.optim.Adam(parameters, lr=config['rqvae_learning_rate'])
        raise ValueError('Unknown RQ-VAE optimizer')
    optimization = config.get('rqvae_optimization','joint')
    if optimization not in ['joint','alternating']:
        raise ValueError('Unknown RQ-VAE optimization mode')
    if optimization == 'alternating':
        codebook_optimizer = make_optimizer([model.codebooks])
        network_optimizer = make_optimizer([*model.encoder.parameters(),*model.decoder.parameters()])
    else:
        optimizer = make_optimizer(model.parameters())
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    diagnostics, interval = [], config.get('rqvae_diagnostics_every',0)
    def record_diagnostics(epoch, steps):
        row = {'epoch':epoch,'steps':steps,
               **reconstruction_diagnostics(model,embeddings,batch_size,device)}
        diagnostics.append(row)
        (output/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
        log({'rqvae_diagnostics':{key:value for key,value in row.items() if key!='usage'},
             'used_codes':[value['used'] for value in row['usage']]})
    if interval:
        record_diagnostics(0,0)
    history, steps = [], 0
    log(f'RQ-VAE: {len(embeddings)} objects; settings={settings}; device={device}; optimization={optimization}')
    for epoch in range(config['rqvae_epochs']):
        model.train()
        indices = rng.permutation(len(embeddings))
        totals = {name:0. for name in ['reconstruction','codebook','commitment','total']}
        seen = 0
        for start in range(0, len(indices), batch_size):
            batch = torch.as_tensor(embeddings[indices[start:start+batch_size]], device=device)
            if optimization == 'alternating':
                update_codebooks(model,batch,codebook_optimizer)
                losses = update_network(model,batch,network_optimizer)
            else:
                optimizer.zero_grad(set_to_none=True)
                _, _, losses = model(batch)
                if not torch.isfinite(losses['total']):
                    raise ValueError('Non-finite RQ-VAE loss')
                losses['total'].backward()
                optimizer.step()
            for name, value in losses.items():
                totals[name] += float(value.detach())*len(batch)
            seen += len(batch); steps += 1
            if config.get('rqvae_max_steps') and steps >= config['rqvae_max_steps']:
                break
        record = {'epoch':epoch+1, 'steps':steps, **{name:value/seen for name,value in totals.items()}}
        history.append(record)
        (output/'learning_curve.json').write_text(json.dumps(history, indent=2))
        if epoch == 0 or (epoch+1) % 10 == 0 or epoch+1 == config['rqvae_epochs']:
            log({'rqvae':record})
        limited = config.get('rqvae_max_steps') and steps >= config['rqvae_max_steps']
        if interval and (epoch == 0 or (epoch+1)%interval == 0 or epoch+1==config['rqvae_epochs'] or limited):
            record_diagnostics(epoch+1,steps)
        if config.get('rqvae_max_steps') and steps >= config['rqvae_max_steps']:
            break
    codes, final_reconstruction = encode_all(model, embeddings, batch_size, device)
    checkpoint = {'settings': settings, 'state_dict':model.state_dict(), 'config':config, 'steps':steps}
    torch.save(checkpoint, output/'model.pt')
    loaded = torch.load(output/'model.pt', map_location=device, weights_only=True)
    restored = RQVAE(**loaded['settings']).to(device)
    restored.load_state_dict(loaded['state_dict'])
    repeated, _ = encode_all(restored, embeddings, batch_size, device)
    if not np.array_equal(codes, repeated):
        raise AssertionError('Reloaded RQ-VAE changed codes')
    usage = [{'centers':config['code_width'], 'used':int(len(np.unique(codes[:,i])))}
             for i in range(codes.shape[1])]
    report = {'algorithm':'rqvae', 'settings':settings, 'steps':steps, 'epochs':len(history),
              'optimization':optimization, 'optimizer_updates':steps*(2 if optimization=='alternating' else 1),
              'network_objective':'reconstruction' if optimization=='alternating' else 'reconstruction + commitment',
              'initial_reconstruction':initial_reconstruction, 'final_reconstruction':final_reconstruction,
              'usage':usage, 'initial_usage':[int(len(np.unique(initial_codes[:,i]))) for i in range(codes.shape[1])],
              'checkpoint_reload':True, 'elapsed_seconds':time.monotonic()-started,
              'limited_steps':bool(config.get('rqvae_max_steps')),
              'selection':'final epoch; no recommendation labels used'}
    (output/'metrics.json').write_text(json.dumps(report, indent=2))
    if any(row['used']/row['centers'] < .8 for row in usage):
        log('RQ-VAE: some codebooks use <80% of codes; inspect tokenizer/metrics.json before interpreting quality.')
    log({'rqvae_complete':report})
    books = [book.detach().cpu().numpy().copy() for book in model.codebooks]
    return codes, books, report
