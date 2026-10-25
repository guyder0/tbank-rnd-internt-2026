"""CPU smoke and logged SID experiments. Run as a module from the repository root."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import resource
import shutil
import sys
import time
import traceback

import numpy as np
import pandas as pd

from baseline.pipeline import ROOT, read_data
from research.generative_recsys.src.data import prepare_queries, synthetic_data
from research.generative_recsys.src.tokenizer import content_embeddings, fit_codes, item_texts, text_fingerprint, validate_prepared_config

HOME = ROOT/'research/generative_recsys'


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024*1024), b''):
            result.update(part)
    return result.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False))


class Tee:
    def __init__(self, console, logfile):
        self.console, self.logfile = console, logfile
    def write(self, text):
        self.console.write(text); self.logfile.write(text)
        self.flush()
    def flush(self):
        self.console.flush(); self.logfile.flush()
    def isatty(self):
        return False


def log(value):
    print(time.strftime('%H:%M:%S'), value, flush=True)


def preflight(device):
    try:
        import torch
    except ImportError as error:
        raise RuntimeError('PyTorch is missing in baseline/.venv. See research/generative_recsys/RUNNING.md; prepare-only and NumPy tests need no torch.') from error
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable. No CPU fallback; run from the host terminal with a CUDA-enabled PyTorch build.')
        probe = torch.ones((16,16), device=device)
        _ = probe@probe
        torch.cuda.synchronize()
    return torch.device(device)


def make_prepared(config, output, args):
    start = time.monotonic()
    if args.smoke and not args.real_data:
        events, items, chosen = synthetic_data()
        source = {'synthetic': True}
    else:
        log('Reading complete Russian KION archive')
        events, _, items = read_data(ROOT/'KION_DATASET')
        chosen_path = ROOT/'baseline/artifacts/default/sample_users.csv'
        chosen = pd.read_csv(chosen_path).user_id.to_numpy()
        if args.max_users or args.smoke:
            chosen = chosen[:args.max_users or 300]
        source = {'synthetic': False, 'archive_sha256': digest(ROOT/'KION_DATASET/data_original.zip'),
                  'items_sha256': digest(ROOT/'KION_DATASET/data_original/items.csv'),
                  'sample_users_sha256': digest(chosen_path), 'selected_users': len(chosen)}
    cutoffs = ['2021-07-19','2021-07-26','2021-08-02','2021-08-09']
    catalog, splits = prepare_queries(events, items, chosen, cutoffs, config['top_k'])
    log(f'Closed catalog: {len(catalog)} items observed before {cutoffs[0]}')
    texts = item_texts(catalog)
    embeddings, embedding_meta = content_embeddings(texts, config, args.device)
    log(f'Embeddings {embeddings.shape}; backend={embedding_meta}')
    codebook = fit_codes(catalog.item_id.to_numpy(), embeddings, config,
                         output=output/'tokenizer', device=args.device, log=log)
    log(f'SID diagnostics: {codebook["diagnostics"]}')
    value = {'codebook': codebook, 'splits': splits, 'config': config, 'source': source}
    with (output/'prepared.pkl').open('wb') as stream:
        pickle.dump(value, stream, protocol=5)
    np.save(output/'item_embeddings.npy', embeddings)
    columns = {f'code_{i}': codebook['codes'][:,i] for i in range(codebook['codes'].shape[1])}
    pd.DataFrame({'item_id': codebook['item_ids'], **columns}).to_csv(output/'semantic_ids.csv', index=False)
    write_json(output/'prepare_manifest.json', {'source': source, 'embedding': embedding_meta,
        'texts_sha256': text_fingerprint(texts), 'config': config,
        'embeddings_sha256':digest(output/'item_embeddings.npy'),
        'prepared_sha256': digest(output/'prepared.pkl'), 'catalog_size': len(catalog),
        'catalog_policy': 'fixed closed catalog observed before earliest train cutoff; out-of-catalog positives retained in evaluation truth',
        'diagnostics': codebook['diagnostics'], 'prepare_seconds': time.monotonic()-start,
        'splits': [{'name':s['name'], 'cutoff':s['cutoff'], 'max_history_date':s['max_history_date'],
                    'queries':len(s['queries']), 'positives':sum(len(q['positives']) for q in s['queries'])} for s in splits]})
    return value


def compare_catboost(prepared, output, report, seed):
    if prepared['source']['synthetic']:
        return
    from research.generative_recsys.src.metrics import paired_delta
    source = ROOT/'baseline/artifacts/strong/expanded/validation_catboost_per_user.csv'
    truth_file = ROOT/'baseline/artifacts/default/validation_2021-08-09.truth.json'
    if not source.exists() or not truth_file.exists():
        log('No saved CatBoost reference; skipped')
        return
    truth = json.loads(truth_file.read_text())
    queries = next(s['queries'] for s in prepared['splits'] if s['name']=='validation')
    if any(set(q['positives']) != set(truth.get(str(q['user_id']), [])) for q in queries):
        raise ValueError('Saved CatBoost reference has different validation truth')
    selected = pd.read_csv(output/'validation_sid_per_user.csv')
    baseline = pd.read_csv(source)
    baseline = baseline[baseline.user_id.isin(selected.user_id)]
    report['catboost_reference'] = {'source': str(source.relative_to(ROOT)), 'sha256': digest(source),
        'ndcg@10': float(baseline.ndcg.mean()),
        'recall@10': float(baseline.recall.mean()),
        'map@10': float(baseline['map'].mean()),
        'comparison': 'whole pipelines: CatBoost uses expanded candidate retrieval; SID generates over fixed earliest-cutoff catalog',
        **paired_delta(selected, baseline, seed)}
    write_json(output/'metrics.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--pretrained-smoke', action='store_true',
                        help='With --smoke, retain configured pretrained embeddings (downloads weights)')
    parser.add_argument('--real-data', action='store_true', help='Use small real-user sample with --smoke')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--prepared', type=Path, help='Existing prepared run directory')
    parser.add_argument('--config', type=Path, default=HOME/'configs/default.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--max-users', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--embedding-backend', choices=['tfidf','e5','sentence_t5'])
    parser.add_argument('--random-codes', action='store_true')
    parser.add_argument('--tokenizer', choices=['rqvae','residual_kmeans'])
    args = parser.parse_args()
    if args.pretrained_smoke and not args.smoke:
        parser.error('--pretrained-smoke requires --smoke')
    if Path(sys.prefix).resolve() != (ROOT/'baseline/.venv').resolve():
        parser.error('Use baseline/.venv/bin/python')
    if args.smoke and args.device != 'cpu':
        parser.error('--smoke must use --device cpu')
    if args.max_users is not None and args.max_users < 1:
        parser.error('--max-users must be positive')
    loaded_config = json.loads(args.config.read_text())
    config = {**json.loads((HOME/'configs/default.json').read_text()), **loaded_config}
    if 'tokenizer' not in loaded_config:
        config['tokenizer'] = 'residual_kmeans'
    if args.embedding_backend:
        config['embedding_backend'] = args.embedding_backend
        if args.embedding_backend == 'tfidf':
            config['embedding_dim'] = 96
        elif args.embedding_backend == 'e5':
            config['embedding_model'] = 'intfloat/multilingual-e5-small'
        elif args.embedding_backend == 'sentence_t5':
            config.update(embedding_model='sentence-transformers/sentence-t5-base', embedding_dim=768)
    if args.tokenizer:
        config['tokenizer'] = args.tokenizer
    if args.random_codes:
        config['random_codes'] = True
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.smoke:
        embedding_settings = {key:value for key,value in config.items() if key.startswith('embedding_')}
        config.update(embedding_backend='tfidf', embedding_dim=8, code_width=4,
                      kmeans_iterations=5, d_model=16, heads=2, layers=1,
                      batch_size=32, epochs=2, beam_width=16, threads=2)
        if args.pretrained_smoke:
            config.update(embedding_settings)
        if config.get('tokenizer') == 'rqvae':
            config.update(rqvae_hidden=[16,8], rqvae_latent_dim=4, rqvae_batch_size=32,
                          rqvae_epochs=3, rqvae_max_steps=4)
    if config['epochs'] < 1 or config['beam_width'] < config['top_k']:
        parser.error('Need positive epochs and beam_width >= top_k')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = (args.output or HOME/'artifacts'/stamp).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'config.json', config)
    sources = output/'source'; sources.mkdir()
    for path in [Path(__file__).resolve(), *sorted((HOME/'src').glob('*.py'))]:
        shutil.copy2(path, sources/path.name)
    status = {'status':'running', 'device':args.device, 'smoke':args.smoke,
              'command':sys.argv, 'started_utc':stamp, 'python':sys.version,
              'source_sha256': {p.name:digest(p) for p in sources.iterdir()},
              'dependencies': {d.metadata['Name']:d.version for d in importlib.metadata.distributions()}}
    started = time.monotonic()
    old_stdout, old_stderr = sys.stdout, sys.stderr
    with (output/'run.log').open('w', buffering=1) as logfile:
        sys.stdout, sys.stderr = Tee(old_stdout, logfile), Tee(old_stderr, logfile)
        try:
            log(f'Run: {output}')
            status['stage']='preflight'; write_json(output/'status.json', status)
            needs_torch = (not args.prepare_only or config.get('tokenizer') == 'rqvae'
                           or config['embedding_backend'] in ['e5','sentence_t5'])
            device = preflight(args.device) if needs_torch else None
            status['stage']='prepare'; write_json(output/'status.json', status)
            if args.prepared:
                directory = args.prepared.resolve()
                manifest = json.loads((directory/'prepare_manifest.json').read_text())
                if digest(directory/'prepared.pkl') != manifest['prepared_sha256']:
                    raise ValueError('Prepared data checksum mismatch')
                with (directory/'prepared.pkl').open('rb') as stream:
                    prepared = pickle.load(stream)
                validate_prepared_config(config, prepared['config'])
                status['prepared_source']=str(directory)
                if args.max_users or args.real_data:
                    raise ValueError('User selection cannot change when reusing prepared data')
            else:
                prepared = make_prepared(config, output, args)
            if args.prepare_only:
                status.update(status='prepared',stage='done')
            else:
                from research.generative_recsys.src.training import train
                status['stage']='train'; write_json(output/'status.json', status)
                report = train(prepared, config, output, device, log, args.smoke)
                compare_catboost(prepared, output, report, config['seed'])
                log({'validation':report['validation'], 'delta_vs_popularity':report['delta_vs_popularity']})
                status.update(status='completed', stage='done')
        except BaseException as error:
            status.update(status='failed', error=repr(error))
            traceback.print_exc()
            raise
        finally:
            status['elapsed_seconds']=time.monotonic()-started
            status['peak_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
            write_json(output/'status.json',status)
            sys.stdout, sys.stderr = old_stdout, old_stderr


if __name__=='__main__':
    main()
