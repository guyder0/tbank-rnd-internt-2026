"""Manual-aggregate experiments: reserve, prepare, train, final.

Run exclusively with baseline/.venv/bin/python -u -m baseline.strong.
"""
import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool

from baseline.aggregates import ManualAggregates
from baseline.pipeline import (ROOT, CATEGORY_COLUMNS, evaluate, history_before,
                               log, make_candidates, make_features, read_data, targets)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False))


def reserve(output, base, config, events):
    path = output / 'holdout_users.csv'
    original = pd.read_csv(base / 'sample_users.csv').user_id.to_numpy()
    if path.exists():
        selected = pd.read_csv(path).user_id.to_numpy()
    else:
        population = np.setdiff1d(events.user_id.unique(), original)
        rng = np.random.default_rng(config['holdout_seed'])
        selected = np.sort(rng.choice(population, config['holdout_users'], replace=False))
        pd.Series(selected, name='user_id').to_csv(path, index=False)
        write_json(output / 'holdout_protocol.json', {
            'seed': config['holdout_seed'], 'users': len(selected),
            'excluded_development_users': len(original),
            'sampling': 'uniform user IDs, no future activity or labels inspected',
            'limitation': 'new users in the previously evaluated calendar week, not a new time period',
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    if np.intersect1d(selected, original).size or len(selected) != config['holdout_users']:
        raise ValueError('Invalid holdout user partition')
    protocol = read_json(output / 'holdout_protocol.json')
    if hashlib.sha256(path.read_bytes()).hexdigest() != protocol['sha256']:
        raise ValueError('Reserved holdout user list was modified')
    return selected


def context(output, config, history, users, items, cutoff, user_ids):
    signature = hashlib.sha256(json.dumps({k: config[k] for k in [
        'neighbors', 'minimum_support', 'similarity_shrinkage', 'graph_history_limit']}, sort_keys=True).encode()).hexdigest()[:10]
    graph_name = f'graph_{cutoff.date()}_{signature}.npz'
    cached = Path(config.get('graph_cache_directory', output)) / graph_name
    graph = cached if cached.exists() else output / graph_name
    return ManualAggregates(history, users, items, cutoff, config, graph_path=graph, user_ids=user_ids)


def prepare(output, base, config, variant, events, users, items):
    started = time.monotonic()
    base_manifest = read_json(base / 'manifest.json')
    chosen = pd.read_csv(base / 'sample_users.csv').user_id.to_numpy()
    if config.get('smoke_users'):
        chosen = chosen[:config['smoke_users']]
    manifest = {'config': config, 'variant': variant, 'splits': [],
                'base_config': base_manifest['config'],
                'evaluation_population': 'warm users with at least one future positive; retrieval misses retained'}
    destination = output / variant
    destination.mkdir(exist_ok=True)
    for index, record in enumerate(base_manifest['splits']):
        if record['split'] == 'test':
            continue
        cutoff = pd.Timestamp(record['cutoff'])
        name = f"{record['split']}_{cutoff.date()}"
        log(f'{variant}: preparing {name}')
        history = history_before(events, cutoff)
        eligible = np.intersect1d(chosen, history.user_id.unique())
        truth = targets(events, eligible, cutoff, 7)
        # Offline metrics are conditional on future activity. This only avoids scoring
        # users absent from the metric; candidate generation still sees past data only.
        query_users = np.array(sorted(truth))
        ctx = context(output, config, history, users, items, cutoff, query_users)
        if variant == 'fixed':
            frame = pd.read_pickle(base / record['path'])
            frame = frame[frame.user_id.isin(query_users)].reset_index(drop=True)
            manual = ctx.features(frame[['user_id', 'item_id']])
            frame = pd.concat([frame, manual], axis=1)
        else:
            candidates = make_candidates(history, query_users, items, base_manifest['config'], cutoff)
            candidates = ctx.expand_candidates(candidates)
            candidates['label'] = [int(item in truth.get(user, set()))
                                   for user, item in candidates.itertuples(index=False, name=None)]
            if record['split'] == 'train':
                candidates = candidates[candidates.groupby('user_id').label.transform('sum').gt(0)]
            frames = []
            batch_users = candidates.user_id.unique()
            step = config['feature_batch_users']
            for offset in range(0, len(batch_users), step):
                batch = candidates[candidates.user_id.isin(batch_users[offset:offset+step])].reset_index(drop=True)
                core = make_features(batch, history, users, items, cutoff, base_manifest['config']['windows'])
                manual = ctx.features(batch[['user_id', 'item_id']])
                frames.append(pd.concat([core, manual], axis=1))
                log(f'  Features {min(offset+step,len(batch_users))}/{len(batch_users)} users')
            frame = pd.concat(frames, ignore_index=True)
            frame['query_id'] = index * (int(events.user_id.max()) + 1) + frame.user_id
        frame = frame.sort_values(['query_id', 'item_id']).reset_index(drop=True)
        if frame.isna().any().any():
            raise ValueError('Missing feature value')
        frame.to_pickle(destination / f'{name}.pkl')
        write_json(destination / f'{name}.truth.json', {str(k): sorted(v) for k,v in truth.items()})
        manifest['splits'].append({'split': record['split'], 'cutoff': str(cutoff.date()),
                                   'path': f'{name}.pkl', 'truth': f'{name}.truth.json',
                                   'rows': len(frame), 'positive_rows': int(frame.label.sum()), 'positive_users': len(truth)})
        log(str(manifest['splits'][-1]))
    manifest['prepare_seconds'] = time.monotonic() - started
    write_json(destination / 'manifest.json', manifest)


def load(destination, manifest, split):
    records = [r for r in manifest['splits'] if r['split'] == split]
    frame = pd.concat([pd.read_pickle(destination / r['path']) for r in records], ignore_index=True)
    frame = frame.sort_values(['query_id', 'item_id']).reset_index(drop=True)
    truth = {int(k): set(v) for k,v in read_json(destination / records[0]['truth']).items()} if split != 'train' else {}
    return frame, truth


def ndcg_fast(frame, scores, truth, k=10):
    order = np.lexsort((frame.item_id.to_numpy(), -np.asarray(scores), frame.user_id.to_numpy()))
    groups = frame.user_id.to_numpy()[order]
    starts = np.r_[0, np.flatnonzero(groups[1:] != groups[:-1]) + 1]
    sizes = np.diff(np.r_[starts, len(groups)])
    ranks = np.arange(len(groups)) - np.repeat(starts, sizes)
    hits = frame.label.to_numpy()[order]
    discounts = hits / np.log2(ranks + 2)
    discounts[ranks >= k] = 0
    dcg = np.add.reduceat(discounts, starts)
    ideal = np.array([np.sum(1/np.log2(np.arange(min(k, len(truth[int(user)])))+2)) for user in groups[starts]])
    return float(np.sum(dcg/ideal) / len(truth))


def compare(frame, predictions, truth, destination, name, seed):
    metrics, individual = {}, {}
    for model, score in predictions.items():
        metrics[model], individual[model] = evaluate(frame, score, truth)
        individual[model].to_csv(destination / f'{name}_{model}_per_user.csv', index=False)
        log(f'{name} {model}: {metrics[model]}')
    primary = 'catboost' if 'catboost' in predictions else list(predictions)[-1]
    for reference in predictions:
        if reference == primary:
            continue
        delta = individual[primary].ndcg.to_numpy() - individual[reference].ndcg.to_numpy()
        rng = np.random.default_rng(seed)
        boot = [float(rng.choice(delta, len(delta), replace=True).mean()) for _ in range(1000)]
        metrics[f'delta_vs_{reference}'] = {'ndcg': float(delta.mean()),
                                           'ci95': np.quantile(boot, [.025,.975]).tolist()}
    return metrics


def select_features(training, config, ablation=None):
    columns = [c for c in training if c not in ['user_id','item_id','label','query_id']]
    allowed = set(columns)
    if config.get('feature_columns') is not None:
        columns = config['feature_columns']
        if len(columns) != len(set(columns)) or not set(columns).issubset(allowed):
            raise ValueError('Invalid explicit feature columns: labels and identifiers are forbidden')
    if ablation == 'no_co':
        columns = [c for c in columns if not c.startswith('manual_co_')]
    return columns


def train(output, base, config, variant, ablation=None):
    destination = output / variant
    manifest = read_json(destination / 'manifest.json')
    if config != manifest['config']:
        raise ValueError('Config differs from prepared aggregate features')
    training, _ = load(destination, manifest, 'train')
    validation, truth = load(destination, manifest, 'validation')
    columns = select_features(training, config, ablation)
    folder = destination if ablation is None else destination / ablation
    folder.mkdir(exist_ok=True)
    def pool(frame):
        return Pool(frame[columns], frame.label, group_id=frame.query_id, cat_features=CATEGORY_COLUMNS)
    log(f'{variant} {ablation or "full"}: fitting {len(training)} rows, {len(columns)} features')
    started = time.monotonic()
    device_params = {'task_type': config.get('task_type', 'CPU')}
    if device_params['task_type'] == 'GPU':
        device_params.update(devices=config.get('devices', '0'),
                             gpu_ram_part=config.get('gpu_ram_part', 0.75))
    model = CatBoostRanker(**device_params, iterations=config['iterations'], depth=config['depth'],
                          learning_rate=config['learning_rate'], l2_leaf_reg=config['l2_leaf_reg'],
                          loss_function='YetiRank', eval_metric='NDCG:top=10',
                          thread_count=config['thread_count'], random_seed=config['seed'], allow_writing_files=False)
    model.fit(pool(training), eval_set=pool(validation), use_best_model=False,
              early_stopping_rounds=100, verbose=50)
    write_json(folder / 'learning_curve.json', model.get_evals_result())
    write_json(folder / 'catboost_parameters.json', model.get_all_params())
    # Select by actual end-to-end NDCG, including retrieval misses and full positives.
    selection, best_score, best_trees = [], -1, 0
    for index, prediction in enumerate(model.staged_predict(validation[columns], eval_period=25)):
        trees = min((index+1)*25, model.tree_count_)
        value = ndcg_fast(validation, prediction, truth)
        selection.append({'trees': trees, 'ndcg@10': value})
        if value > best_score:
            best_score, best_trees = value, trees
    model.shrink(best_trees)
    model.save_model(str(folder / 'model.cbm'))
    log(f'End-to-end validation selection: {best_trees} trees, NDCG={best_score:.6f}')
    old = CatBoostRanker().load_model(str(base / 'model.cbm'))
    predictions = {'popularity': validation.item_count_30d.to_numpy(),
                   'item_to_item': validation.manual_co_sum.to_numpy(),
                   'previous_catboost': old.predict(validation[old.feature_names_]),
                   'catboost': model.predict(validation[columns])}
    reference_path = config.get('reference_model')
    if reference_path:
        reference = CatBoostRanker().load_model(str(reference_path))
        predictions['strong_v2'] = reference.predict(validation[reference.feature_names_])
    metrics = compare(validation, predictions, truth, folder, 'validation', config['seed'])
    importance = model.get_feature_importance(type='PredictionValuesChange')
    pd.DataFrame({'feature': columns, 'importance': importance}).sort_values('importance', ascending=False).to_csv(folder / 'feature_importance.csv', index=False)
    write_json(folder / 'metrics.json', {'config': config, 'variant': variant, 'ablation': ablation,
               'train_seconds': time.monotonic()-started, 'trees': best_trees, 'features': columns,
               'selection': selection, 'validation': metrics,
               'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024})


def final(output, base, config, events, users, items):
    started = time.monotonic()
    destination = output / 'final'
    if destination.exists():
        raise ValueError('Final evaluation already exists; do not repeatedly inspect holdout')
    # Freeze model checksums BEFORE accessing holdout outcomes.
    paths = {name: output / name / 'model.cbm' for name in ['fixed','expanded']}
    paths['no_co'] = output / 'fixed/no_co/model.cbm'
    paths['previous'] = base / 'model.cbm'
    checksums = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name,path in paths.items()}
    validation_scores = {name: read_json(path.parent / 'metrics.json')['validation']['catboost']['ndcg@10']
                         for name,path in paths.items() if name != 'previous'}
    selected = max(validation_scores, key=validation_scores.get)
    chosen = reserve(output, base, config, events)
    destination.mkdir()
    write_json(destination / 'frozen_models.json', checksums)
    write_json(output / 'selection.json', {'primary_variant': selected, 'validation_ndcg': validation_scores,
                                         'model_sha256': checksums[selected],
                                         'model_path': str(paths[selected].relative_to(output)),
                                         'criterion': 'validation only'})
    cutoff = events.date.max() + pd.Timedelta(days=1) - pd.Timedelta(days=7)
    history = history_before(events, cutoff)
    eligible = np.intersect1d(chosen, history.user_id.unique())
    truth = targets(events, eligible, cutoff, 7)
    base_manifest = read_json(base / 'manifest.json')
    ctx = context(output, config, history, users, items, cutoff, sorted(truth))
    original = make_candidates(history, sorted(truth), items, base_manifest['config'], cutoff)
    expanded = ctx.expand_candidates(original)
    models = {name: CatBoostRanker().load_model(str(path)) for name,path in paths.items()}
    report = {'cutoff': str(cutoff.date()), 'users': len(truth), 'frozen_models': checksums,
              'primary_variant': selected,
              'candidate_rows': {'fixed': len(original), 'expanded': len(expanded)},
              'limitation': 'disjoint users in same known week; not a fresh temporal test'}
    for variant, candidate in [('fixed',original),('expanded',expanded)]:
        scored_parts = []
        ids = candidate.user_id.unique()
        for offset in range(0,len(ids),config['feature_batch_users']):
            batch = candidate[candidate.user_id.isin(ids[offset:offset+config['feature_batch_users']])].reset_index(drop=True)
            core = make_features(batch, history, users, items, cutoff, base_manifest['config']['windows'])
            frame = pd.concat([core, ctx.features(batch)], axis=1)
            scored = batch.copy()
            scored['popularity'] = frame.item_count_30d.to_numpy()
            scored['item_to_item'] = frame.manual_co_sum.to_numpy()
            scored['previous_catboost'] = models['previous'].predict(frame[models['previous'].feature_names_])
            scored['catboost'] = models[variant].predict(frame[models[variant].feature_names_])
            if variant == 'fixed':
                scored['no_co'] = models['no_co'].predict(frame[models['no_co'].feature_names_])
            scored_parts.append(scored)
            log(f'Final {variant}: {min(offset+config["feature_batch_users"],len(ids))}/{len(ids)} users')
        scores = pd.concat(scored_parts,ignore_index=True)
        scores.to_pickle(destination / f'{variant}_scores.pkl')
        report[variant] = compare(scores, {n:scores[n].to_numpy() for n in ['popularity','item_to_item','previous_catboost','catboost']},
                                  truth,destination,variant,config['seed'])
        if variant == 'fixed':
            ablation = {name: scores[name].to_numpy() for name in ['popularity','item_to_item','previous_catboost']}
            ablation['catboost'] = scores.no_co.to_numpy()
            report['no_co'] = compare(scores,ablation,truth,destination,'no_co',config['seed'])
    previous = pd.read_csv(destination / 'fixed_previous_catboost_per_user.csv').set_index('user_id')
    primary = pd.read_csv(destination / f'{selected}_catboost_per_user.csv').set_index('user_id')
    difference = (primary.ndcg - previous.ndcg).to_numpy()
    if np.isnan(difference).any():
        raise ValueError('Holdout comparisons have different user populations')
    rng = np.random.default_rng(config['seed'])
    bootstrap = [float(rng.choice(difference,len(difference),replace=True).mean()) for _ in range(1000)]
    report['primary_vs_original_pipeline'] = {'ndcg_delta': float(difference.mean()),
                                             'ci95': np.quantile(bootstrap,[.025,.975]).tolist()}
    report['elapsed_seconds'] = time.monotonic() - started
    report['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    write_json(destination / 'metrics.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['reserve','prepare','train','final'])
    parser.add_argument('--variant',choices=['fixed','expanded'],default='fixed')
    parser.add_argument('--ablation',choices=['no_co'])
    parser.add_argument('--config',type=Path,default=ROOT/'baseline/configs/strong.json')
    parser.add_argument('--output',type=Path,default=ROOT/'baseline/artifacts/strong')
    parser.add_argument('--base',type=Path,default=ROOT/'baseline/artifacts/default')
    args=parser.parse_args(); config=read_json(args.config)
    args.output.mkdir(parents=True,exist_ok=True)
    if args.command == 'train':
        train(args.output,args.base,config,args.variant,args.ablation)
        return
    events,users,items=read_data(ROOT/'KION_DATASET')
    reserve(args.output,args.base,config,events)
    if args.command == 'prepare': prepare(args.output,args.base,config,args.variant,events,users,items)
    if args.command == 'final': final(args.output,args.base,config,events,users,items)


if __name__ == '__main__':
    main()
