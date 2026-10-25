"""Content embeddings and selectable residual k-means / TIGER-style RQ-VAE."""
from collections import Counter
import hashlib
import re
import warnings

import numpy as np
from scipy import sparse
from scipy.cluster.vq import kmeans2
from scipy.sparse.linalg import svds

PAD, BOS, UNK = 0, 1, 2


def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def item_texts(items):
    fields = ['title', 'genres', 'countries', 'directors', 'actors', 'content_type']
    return ['; '.join(f'{field}: {str(row[field])}' for field in fields
                     if field in items and str(row[field]) not in ['nan', '<NA>', ''])
            for _, row in items.iterrows()]


def lexical_embeddings(texts, dimensions, seed):
    """Lexical TF-IDF + truncated SVD control, not pretrained language semantics."""
    documents = [Counter(re.findall(r'[\w]+', text.lower())) for text in texts]
    frequency = Counter(token for document in documents for token in document)
    vocabulary = sorted(frequency, key=lambda token: (-frequency[token], token))[:16000]
    if not vocabulary:
        raise ValueError('Empty content vocabulary')
    indices = {token: i for i, token in enumerate(vocabulary)}
    rows, cols, values = [], [], []
    for row, counts in enumerate(documents):
        for token, count in counts.items():
            if token in indices:
                rows.append(row); cols.append(indices[token])
                values.append((1 + np.log(count)) * (1 + np.log((1+len(texts))/(1+frequency[token]))))
    matrix = sparse.csr_matrix((values, (rows, cols)), shape=(len(texts), len(vocabulary)), dtype='float32')
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    matrix = sparse.diags(1 / np.maximum(norms, 1e-8)) @ matrix
    rank = min(dimensions, min(matrix.shape)-1)
    if rank < 1:
        result = matrix.toarray()
    else:
        u, singular, _ = svds(matrix, k=rank, random_state=seed)
        result = u * singular
    return normalize(np.asarray(result, dtype='float32'))


def content_embeddings(texts, config, device='cpu'):
    if config['embedding_backend'] == 'tfidf':
        return lexical_embeddings(texts, config['embedding_dim'], config['seed']), {'backend': 'tfidf_svd'}
    if config['embedding_backend'] != 'e5':
        raise ValueError('Unknown embedding backend')
    import torch
    from transformers import AutoModel, AutoTokenizer
    kwargs = {'revision': config['embedding_revision']}
    tokenizer = AutoTokenizer.from_pretrained(config['embedding_model'], **kwargs)
    model = AutoModel.from_pretrained(config['embedding_model'], **kwargs).to(device).eval()
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(texts), 16):
            inputs = tokenizer(['query: '+t for t in texts[start:start+16]], padding=True,
                               truncation=True, max_length=128, return_tensors='pt').to(device)
            hidden = model(**inputs).last_hidden_state
            mask = inputs.attention_mask.unsqueeze(-1)
            pooled = (hidden*mask).sum(1)/mask.sum(1).clamp(min=1)
            vectors.append(torch.nn.functional.normalize(pooled, dim=1).cpu().numpy())
    revision = getattr(model.config, '_commit_hash', None)
    return np.concatenate(vectors), {'backend': 'e5', 'model': config['embedding_model'],
                                      'requested_revision': config['embedding_revision'], 'resolved_revision': revision}


def nearest(points, centers):
    output = []
    for start in range(0, len(points), 2048):
        x = points[start:start+2048]
        distance = (x*x).sum(1)[:, None] + (centers*centers).sum(1)[None, :] - 2*x@centers.T
        output.append(distance.argmin(1))
    return np.concatenate(output)


def fit_codes(item_ids, embeddings, config, output=None, device='cpu', log=print):
    """Fit on the fixed earliest-cutoff catalog; suffix makes every SID unique."""
    if len(item_ids) != len(embeddings) or len(set(item_ids)) != len(item_ids):
        raise ValueError('Invalid item/embedding alignment')
    if len(item_ids) == 0 or not np.isfinite(embeddings).all():
        raise ValueError('Nonempty finite embeddings required')
    algorithm = config.get('tokenizer', 'residual_kmeans')
    if algorithm == 'rqvae':
        if output is None:
            raise ValueError('RQ-VAE requires an output directory for checkpoints and logs')
        from research.generative_recsys.src.rqvae import train_rqvae
        raw, centroids, report = train_rqvae(embeddings, config, output, device, log)
        return finalize_codes(item_ids, raw, centroids, config, report)
    if algorithm != 'residual_kmeans':
        raise ValueError('Unknown tokenizer')
    residual = normalize(embeddings.copy())
    levels, centroids, usage = [], [], []
    for level in range(config['code_levels']):
        residual = normalize(residual)
        count = min(config['code_width'], len(np.unique(residual, axis=0)))
        count = max(1, count)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='One of the clusters is empty')
            centers, _ = kmeans2(residual, count, iter=config['kmeans_iterations'],
                                 minit='++', seed=config['seed']+level)
        codes = nearest(residual, centers)
        levels.append(codes); centroids.append(centers)
        usage.append({'centers': len(centers), 'used': int(len(np.unique(codes)))})
        residual = residual-centers[codes]
    raw = np.column_stack(levels)
    return finalize_codes(item_ids, raw, centroids, config, {'usage':usage, 'algorithm':'residual_kmeans'})


def finalize_codes(item_ids, raw, centroids, config, diagnostics):
    suffix, buckets = [], Counter()
    for code in raw:
        key = tuple(code); suffix.append(buckets[key]); buckets[key] += 1
    suffix_width = max(suffix)+1
    codes = np.column_stack([raw, suffix]).astype('int64')
    for level in range(config['code_levels']):
        codes[:, level] += 3+level*config['code_width']
    codes[:, -1] += 3+config['code_levels']*config['code_width']
    if config.get('random_codes'):
        codes = codes[np.random.default_rng(config['seed']).permutation(len(codes))]
    if len({tuple(code) for code in codes}) != len(codes):
        raise AssertionError('SID collisions after suffix')
    return {'item_ids': np.asarray(item_ids, dtype='int64'), 'codes': codes,
            'centroids': centroids, 'vocab_size': 3+config['code_levels']*config['code_width']+suffix_width,
            'tokenizer': config.get('tokenizer', 'residual_kmeans'),
            'diagnostics': {**diagnostics, 'unique_before_suffix': len(buckets),
                            'items': len(codes), 'max_collision_bucket': suffix_width,
                            'random_codes': config.get('random_codes', False)}}


def text_fingerprint(texts):
    return hashlib.sha256('\n'.join(texts).encode()).hexdigest()


TOKENIZER_KEYS = ['embedding_backend', 'embedding_model', 'embedding_revision',
                  'embedding_dim', 'code_width', 'code_levels', 'kmeans_iterations',
                  'random_codes', 'seed']
RQVAE_KEYS = ['rqvae_hidden', 'rqvae_latent_dim', 'rqvae_beta', 'rqvae_batch_size',
              'rqvae_epochs', 'rqvae_optimizer', 'rqvae_learning_rate', 'rqvae_max_steps']


def validate_prepared_config(config, prepared_config):
    algorithm = config.get('tokenizer', 'residual_kmeans')
    if algorithm != prepared_config.get('tokenizer', 'residual_kmeans'):
        raise ValueError('Prepared tokenizer differs: old k-means SID cannot be used as RQ-VAE SID. Prepare a new run.')
    keys = TOKENIZER_KEYS + (RQVAE_KEYS if algorithm == 'rqvae' else [])
    if any(config.get(key) != prepared_config.get(key) for key in keys):
        raise ValueError('Prepared tokenizer config differs from requested config')
