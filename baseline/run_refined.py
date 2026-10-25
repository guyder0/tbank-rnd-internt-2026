"""Controlled manual-feature ablations on frozen candidates; no holdout access."""
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import traceback
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np
from catboost import CatBoostRanker

from baseline.pipeline import ROOT, history_before, make_candidates, read_data
from baseline.refined import RefinedFeatures
from baseline.strong import context, read_json, train, write_json


DUPLICATES = {'manual_user_history_size'}


def feature_sets(columns, v2):
    clean = [c for c in v2 if c not in DUPLICATES]
    groups = {name: [c for c in columns if c.startswith(prefix)] for name, prefix in
              [('sources', 'ref_source_'), ('interest', 'ref_interest_'), ('lift', 'ref_lift_')]}
    return {'v2_gpu': v2, 'v3_gpu': [c for c in columns if not c.startswith('ref_')],
            'clean': clean, **{name: clean + extra for name, extra in groups.items()},
            'combined': clean + sum(groups.values(), [])}


def prepare(source, output, smoke):
    source_manifest = read_json(source/'expanded/manifest.json')
    config = source_manifest['config']
    events, users, items = read_data(ROOT/'KION_DATASET')
    destination = output/'prepared'; destination.mkdir()
    manifest = {**source_manifest, 'splits': [], 'source': str(source),
                'frozen_candidates': True, 'smoke': smoke}
    for record in source_manifest['splits']:
        if record['split'] == 'test':
            raise ValueError('Only development splits are permitted')
        frame = pd.read_pickle(source/'expanded'/record['path'])
        truth = read_json(source/'expanded'/record['truth'])
        if smoke:
            selected = sorted(frame.user_id.unique())[:30]
            frame = frame[frame.user_id.isin(selected)].reset_index(drop=True)
            truth = {k: v for k, v in truth.items() if int(k) in selected}
        cutoff = pd.Timestamp(record['cutoff'])
        history = history_before(events, cutoff)
        ids = sorted(frame.user_id.unique())
        ctx = context(output, config, history, users, items, cutoff, ids)
        original = make_candidates(history, ids, items, source_manifest['base_config'], cutoff)
        builder = RefinedFeatures(ctx, original)
        blocks = []
        for offset in range(0, len(ids), config['feature_batch_users']):
            batch = frame[frame.user_id.isin(ids[offset:offset+config['feature_batch_users']])].reset_index(drop=True)
            extra = builder.features(batch[['user_id', 'item_id']])
            blocks.append(pd.concat([batch, extra], axis=1))
            print(f'{record["path"]}: {min(offset+config["feature_batch_users"],len(ids))}/{len(ids)} users', flush=True)
        refined = pd.concat(blocks, ignore_index=True)
        pd.testing.assert_frame_equal(frame.reset_index(drop=True), refined[frame.columns].reset_index(drop=True))
        if refined.isna().any().any():
            raise ValueError('Missing refined feature')
        refined.to_pickle(destination/record['path'])
        write_json(destination/record['truth'], truth)
        manifest['splits'].append({**record, 'rows': len(refined),
                                  'positive_rows': int(refined.label.sum()), 'positive_users': len(truth)})
    write_json(destination/'manifest.json', manifest)


def execute(args, output):
    source = args.source.resolve()
    if not args.prepare_only:
        # Backend must work before expensive preparation; no silent CPU fallback.
        probe = CatBoostRanker(iterations=2, depth=2, task_type=args.device,
                               thread_count=2, verbose=False, allow_writing_files=False,
                               **({'devices': args.devices} if args.device == 'GPU' else {}))
        probe.fit([[0],[1],[0],[1]], [0,1,0,1], group_id=[0,0,1,1])
    prepared = output/'prepared'
    if not prepared.exists():
        prepare(source, output, args.smoke)
    manifest = read_json(prepared/'manifest.json')
    if manifest['source'] != str(source) or manifest['smoke'] != args.smoke:
        raise ValueError('Prepared cache does not match requested source or smoke mode')
    if args.prepare_only:
        return
    sample = pd.read_pickle(prepared/manifest['splits'][0]['path'])
    columns = [c for c in sample if c not in ['user_id','item_id','label','query_id']]
    v2 = CatBoostRanker().load_model(str(ROOT/'baseline/artifacts/strong/expanded/model.cbm')).feature_names_
    sets = feature_sets(columns, v2)
    rows = []
    for seed in args.seeds:
        for name, selected in sets.items():
            experiment = output/f'{name}_seed{seed}'
            metrics_path = experiment/'expanded/metrics.json'
            config = {**manifest['config'], 'task_type': args.device, 'devices': args.devices,
                      'seed': seed, 'feature_columns': selected}
            if args.smoke:
                config.update(iterations=3, thread_count=2)
            if metrics_path.exists():
                metrics = read_json(metrics_path)
                if metrics['config'] != config:
                    raise ValueError('Existing experiment config differs')
            else:
                folder = experiment/'expanded'; folder.mkdir(parents=True, exist_ok=True)
                for record in manifest['splits']:
                    for key in ['path', 'truth']:
                        target = folder/record[key]
                        if not target.exists():
                            os.link(prepared/record[key], target)
                write_json(folder/'manifest.json', {**manifest, 'config': config})
                write_json(experiment/'config.json', config)
                print(f'Experiment {name}, seed {seed}, {len(selected)} features', flush=True)
                train(experiment, ROOT/'baseline/artifacts/default', config, 'expanded')
                metrics = read_json(metrics_path)
            rows.append({'experiment': name, 'seed': seed, 'features': len(selected),
                         'trees': metrics['trees'], 'seconds': metrics['train_seconds'],
                         **metrics['validation']['catboost']})
            pd.DataFrame(rows).to_csv(output/'summary.csv', index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    table = pd.DataFrame(rows)
    table.groupby('experiment')[['ndcg@10','recall@10','map@10']].agg(['mean','std']).to_csv(output/'aggregate.csv')
    comparisons = []
    for seed in args.seeds:
        for name in sets:
            candidate = pd.read_csv(output/f'{name}_seed{seed}/expanded/validation_catboost_per_user.csv').set_index('user_id').sort_index()
            for reference_name in ['v2_gpu', 'clean']:
                if name == reference_name:
                    continue
                reference = pd.read_csv(output/f'{reference_name}_seed{seed}/expanded/validation_catboost_per_user.csv').set_index('user_id').sort_index()
                if not candidate.index.equals(reference.index):
                    raise ValueError('Different validation user populations')
                delta = (candidate.ndcg-reference.ndcg).to_numpy()
                rng = np.random.default_rng(seed)
                boot = [rng.choice(delta, len(delta), replace=True).mean() for _ in range(1000)]
                comparisons.append({'experiment': name, 'reference': reference_name, 'seed': seed,
                                    'ndcg_delta': float(delta.mean()), 'ci95': np.quantile(boot,[.025,.975]).tolist()})
    write_json(output/'comparisons.json', comparisons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'baseline/artifacts/runs/20261023T085724_746626Z')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', choices=['GPU', 'CPU'], default='GPU')
    parser.add_argument('--devices', default='0')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42,43,44])
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if Path(sys.prefix).resolve() != (ROOT/'baseline/.venv').resolve():
        parser.error('Use baseline/.venv/bin/python')
    if args.smoke:
        args.device = 'CPU'; args.seeds = [42]
    output = (args.output or ROOT/'baseline/artifacts/refined'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_dir = output/'source'
    if source_dir.exists():
        for path in source_dir.iterdir():
            if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256((ROOT/'baseline'/path.name).read_bytes()).digest():
                raise ValueError('Source code changed since this run; use a fresh output directory')
    else:
        source_dir.mkdir()
        for name in ['pipeline.py','aggregates.py','strong.py','refined.py','run_refined.py']:
            shutil.copy2(ROOT/'baseline'/name, source_dir/name)
    status = {'status': 'running', 'device': args.device, 'smoke': args.smoke,
              'python': sys.version, 'platform': platform.platform(),
              'command': sys.argv, 'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_dir.iterdir()}}
    started = time.monotonic()
    write_json(output/'status.json', status)
    # Tee all output, including CatBoost diagnostics, to a persistent log.
    if os.environ.get('REFINED_LOG_CHILD') != '1':
        with (output/'run.log').open('a', buffering=1) as logfile:
            command = [sys.executable, '-u', '-m', 'baseline.run_refined', *sys.argv[1:]]
            if args.output is None:
                command += ['--output', str(output)]
            child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, env={**os.environ, 'REFINED_LOG_CHILD': '1'})
            try:
                for line in child.stdout:
                    print(line, end='', flush=True); logfile.write(line)
                code = child.wait()
            except BaseException:
                child.terminate(); child.wait(); raise
            if code:
                raise SystemExit(code)
        return
    try:
        execute(args, output)
        status['status'] = 'prepared' if args.prepare_only else 'completed'
    except BaseException as error:
        status.update(status='failed', error=repr(error)); traceback.print_exc(); raise
    finally:
        status['elapsed_seconds'] = time.monotonic()-started
        write_json(output/'status.json', status)


if __name__ == '__main__':
    main()
