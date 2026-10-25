"""Training, checkpoint verification and catalog-wide generative evaluation."""
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from research.generative_recsys.src.metrics import evaluate_lists, paired_delta
from research.generative_recsys.src.decoding import SIDTrie, beam_search, exact_search
from research.generative_recsys.src.model import SIDGenerator, decoder_callback, history_batch


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False))



def train(prepared, config, destination, device, log, smoke=False):
    codebook = prepared['codebook']
    mapping = {int(item): code for item, code in zip(codebook['item_ids'], codebook['codes'])}
    trie = SIDTrie(codebook['item_ids'], codebook['codes'])
    queries = [q for split in prepared['splits'] if split['name']=='train' for q in split['queries']]
    validation = next(s['queries'] for s in prepared['splits'] if s['name']=='validation')
    examples = []
    usable_queries = 0
    for query in queries:
        positives = [item for item in query['positives'] if item in mapping]
        if positives:
            usable_queries += 1
            examples.extend((query, item, 1/len(positives)) for item in positives)
    if not examples:
        raise ValueError('No in-catalog training positives')
    scale = len(examples)/usable_queries
    torch.manual_seed(config['seed'])
    torch.set_num_threads(config['threads'])
    model = SIDGenerator(codebook['vocab_size'], codebook['codes'].shape[1], config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                  weight_decay=config['weight_decay'])
    rng = np.random.default_rng(config['seed'])
    catalog = set(mapping)
    coverage = float(np.mean([len(set(q['positives']) & catalog)/len(q['positives']) for q in validation]))
    references, per_user = {}, {}
    for name, field in [('popularity','popularity'), ('popularity_catalog','popularity_catalog')]:
        references[name], per_user[name] = evaluate_lists(validation, [q[field] for q in validation], config['top_k'])
        per_user[name].to_csv(destination/f'validation_{name}_per_user.csv', index=False)
    best, stale, curve = -1., 0, []
    log(f'Training {len(examples)} pairs / {usable_queries} queries; validation {len(validation)} users; catalog {len(mapping)}')
    started = time.monotonic()
    total_steps = 0
    for epoch in range(config['epochs']):
        model.train(); loss_sum = 0.; n_examples = 0
        for start in range(0, len(examples), config['batch_size']):
            if start == 0:
                indices = rng.permutation(len(examples))
            batch = [examples[index] for index in indices[start:start+config['batch_size']]]
            inputs = history_batch([row[0] for row in batch], mapping, model.sid_length, config['max_history'], device)
            targets = torch.tensor(np.stack([mapping[row[1]] for row in batch]), device=device)
            weights = torch.tensor([row[2]*scale for row in batch], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(*inputs, targets)
            losses = F.cross_entropy(logits.transpose(1,2), targets, reduction='none').mean(1)
            loss = (losses*weights).mean()
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total_steps += 1; loss_sum += float(loss.detach())*len(batch); n_examples += len(batch)
            if total_steps % 50 == 0:
                log(f'epoch={epoch+1} step={total_steps} loss={float(loss.detach()):.4f}')
        model.eval()
        generation_start = time.monotonic()
        results = []
        with torch.inference_mode():
            for number, query in enumerate(validation):
                callback = decoder_callback(model, query, mapping, config, device)
                result = beam_search(callback, trie, config['beam_width'], config['top_k'], query['items'])
                results.append([item for item, _ in result])
                if (number+1) % 500 == 0:
                    log(f'Validation generation {number+1}/{len(validation)}')
        metrics, individual = evaluate_lists(validation, results, config['top_k'])
        record = {'epoch': epoch+1, 'loss': loss_sum/n_examples, 'validation': metrics,
                  'generation_seconds': time.monotonic()-generation_start}
        curve.append(record); write_json(destination/'learning_curve.json', curve)
        log(record)
        if metrics['ndcg@10'] > best:
            best, stale = metrics['ndcg@10'], 0
            torch.save({'state_dict': model.state_dict(), 'config': config,
                        'vocab_size': codebook['vocab_size'], 'sid_length': model.sid_length,
                        'epoch': epoch+1}, destination/'model.pt')
            individual.to_csv(destination/'validation_sid_per_user.csv', index=False)
            write_json(destination/'recommendations.json', {str(q['user_id']): r for q,r in zip(validation, results)})
        else:
            stale += 1
        if stale >= config['patience']:
            break
    checkpoint = torch.load(destination/'model.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['state_dict']); model.eval()
    # Every run verifies a saved/reloaded model and scores the same query consistently.
    with torch.inference_mode():
        callback = decoder_callback(model, validation[0], mapping, config, device)
        beam = beam_search(callback, trie, config['beam_width'], config['top_k'], validation[0]['items'])
        reload = SIDGenerator(codebook['vocab_size'], model.sid_length, config).to(device).eval()
        reload.load_state_dict(checkpoint['state_dict'])
        other = decoder_callback(reload, validation[0], mapping, config, device)
        again = beam_search(other, trie, config['beam_width'], config['top_k'], validation[0]['items'])
        if [x[0] for x in beam] != [x[0] for x in again]:
            raise AssertionError('Checkpoint reload changed recommendations')
        search_check = {'checkpoint_reload': True}
        if smoke or len(mapping) <= 128:
            exact = exact_search(callback, trie, config['top_k'], validation[0]['items'])
            search_check['first_query_beam_exact_overlap'] = len(set(x[0] for x in exact)&set(x[0] for x in beam))/max(1,len(exact))
    selected = pd.read_csv(destination/'validation_sid_per_user.csv')
    report = {'best_epoch': checkpoint['epoch'], 'best_ndcg@10': best,
              'validation': {**references, 'sid': curve[checkpoint['epoch']-1]['validation']},
              'delta_vs_popularity': paired_delta(selected, per_user['popularity'], config['seed']),
              'catalog_recall_ceiling': coverage, 'training_pairs': len(examples),
              'training_queries': usable_queries, 'train_seconds': time.monotonic()-started,
              'parameters': sum(p.numel() for p in model.parameters()), 'search_checks': search_check,
              'short_list_users': sum(len(v)<config['top_k'] for v in json.loads((destination/'recommendations.json').read_text()).values()),
              'device': str(device), 'smoke': smoke,
              'peak_cuda_mib': torch.cuda.max_memory_allocated(device)/1024**2 if device.type=='cuda' else None}
    write_json(destination/'metrics.json', report)
    return report
