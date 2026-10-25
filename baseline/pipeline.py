"""Run from the repository root: python -m baseline.pipeline --help."""
from __future__ import annotations

import argparse
import json
import resource
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# Keep local caches readable without an optional Arrow installation.
pd.options.mode.string_storage = 'python'

ROOT = Path(__file__).resolve().parents[1]
CATEGORY_COLUMNS = ['age', 'income', 'sex', 'kids_flg', 'content_type', 'age_rating']
RELATIONS = ['genres', 'countries', 'directors']


def log(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)


def read_data(data_dir):
    """The standalone interactions archive is incomplete; use data_original.zip."""
    with zipfile.ZipFile(data_dir / 'data_original.zip') as archive:
        with archive.open('data_original/interactions.csv') as stream:
            events = pd.read_csv(stream, usecols=['user_id', 'item_id', 'last_watch_dt'])
    events['date'] = pd.to_datetime(events.pop('last_watch_dt'), errors='raise')
    if events[['user_id', 'item_id', 'date']].isna().any().any():
        raise ValueError('Missing event identifiers or dates')
    if events.duplicated(['user_id', 'item_id']).any():
        raise ValueError('Expected one aggregate per user-item pair')
    users = pd.read_csv(data_dir / 'data_original/users.csv')
    items = pd.read_csv(data_dir / 'data_original/items.csv')
    if users.user_id.duplicated().any() or items.item_id.duplicated().any():
        raise ValueError('Duplicate metadata identifiers')
    if not events.item_id.isin(items.item_id).all():
        raise ValueError('Event item absent from metadata')
    return events, users, items


def history_before(events, cutoff):
    return events.loc[events.date < cutoff].copy()


def token_table(items, column):
    result = items[['item_id', column]].copy()
    result['token'] = result[column].fillna('').str.split(',')
    result = result[['item_id', 'token']].explode('token')
    result['token'] = result.token.str.strip()
    return result[result.token.ne('')].drop_duplicates()


def make_candidates(history, user_ids, items, config, cutoff):
    """Candidate generation never reads labels or future interactions."""
    recent = history[history.date >= cutoff - pd.Timedelta(days=30)]
    counts = recent.groupby('item_id').size().sort_values(ascending=False)
    fallback = history.groupby('item_id').size().sort_values(ascending=False)
    order = list(dict.fromkeys([*counts.index, *fallback.index]))
    global_items = order[:config['popular_candidates']]
    genre_tokens = token_table(items, 'genres')
    user_genres = history[history.user_id.isin(user_ids)].merge(genre_tokens, on='item_id')
    preferences = user_genres.groupby(['user_id', 'token']).size().rename('n').reset_index()
    preferences = preferences.sort_values(['n', 'token'], ascending=[False, True])
    top_genres = preferences.groupby('user_id').head(3).groupby('user_id').token.agg(list).to_dict()
    genre_pop = recent.merge(genre_tokens, on='item_id').groupby(['token', 'item_id']).size()
    genre_items = {}
    if len(genre_pop):
        gp = genre_pop.rename('n').reset_index().sort_values(['n', 'item_id'], ascending=[False, True])
        genre_items = gp.groupby('token').item_id.agg(list).to_dict()
    seen = history[history.user_id.isin(user_ids)].groupby('user_id').item_id.agg(set).to_dict()
    rows = []
    for user in sorted(user_ids):
        personalized = []
        for genre in top_genres.get(user, []):
            personalized.extend(genre_items.get(genre, [])[:config['genre_candidates'] // 3])
        for item in sorted(set(global_items + personalized) - seen.get(user, set())):
            rows.append((user, item))
    return pd.DataFrame(rows, columns=['user_id', 'item_id'])


def make_features(candidates, history, users, items, cutoff, windows):
    """Every dynamic feature is computed exclusively from history before cutoff."""
    if len(history) and history.date.max() >= cutoff:
        raise ValueError('Future event passed to feature builder')
    frame = candidates.copy()
    local = history[history.user_id.isin(frame.user_id.unique())]
    for entity, source in [('user_id', local), ('item_id', history)]:
        prefix = entity.removesuffix('_id')
        stats = source.groupby(entity).agg(**{
            f'{prefix}_count_all': ('date', 'size'),
            f'{prefix}_last_date': ('date', 'max'),
            f'{prefix}_first_date': ('date', 'min'),
        })
        for kind in ['last', 'first']:
            stats[f'{prefix}_{kind}_days'] = (cutoff - stats.pop(f'{prefix}_{kind}_date')).dt.days
        for days in windows:
            selected = source[source.date >= cutoff - pd.Timedelta(days=days)]
            stats[f'{prefix}_count_{days}d'] = selected.groupby(entity).size()
        frame = frame.merge(stats, on=entity, how='left', validate='many_to_one')
        frame[f'{prefix}_trend_7_30'] = ((frame[f'{prefix}_count_7d'].fillna(0) + 1) /
                                                      (frame[f'{prefix}_count_30d'].fillna(0) + 4))
    users = users.copy()
    for column in CATEGORY_COLUMNS:
        if column in users:
            users[column] = users[column].astype('string').fillna('unknown').astype(str)
    frame = frame.merge(users, on='user_id', how='left', validate='many_to_one')
    metadata = items[['item_id', 'content_type', 'release_year', 'age_rating', 'for_kids']].copy()
    for column in ['content_type', 'age_rating']:
        metadata[column] = metadata[column].astype('string').fillna('unknown').astype(str)
    frame = frame.merge(metadata, on='item_id', how='left', validate='many_to_one')
    frame['item_age_years'] = cutoff.year - frame.release_year
    for relation in RELATIONS:
        tokens = token_table(items, relation)
        expanded = local.merge(tokens, on='item_id')
        pair_tokens = frame[['user_id', 'item_id']].merge(tokens, on='item_id')
        for days in [None, *windows]:
            subset = expanded if days is None else expanded[expanded.date >= cutoff - pd.Timedelta(days=days)]
            suffix = 'all' if days is None else f'{days}d'
            affinity = subset.groupby(['user_id', 'token']).size().rename('count').reset_index()
            matched = pair_tokens.merge(affinity, on=['user_id', 'token'], how='left')
            stats = matched.groupby(['user_id', 'item_id'])['count'].agg(['sum', 'max', 'mean'])
            stats.columns = [f'affinity_{relation}_{suffix}_{c}' for c in stats.columns]
            frame = frame.merge(stats, on=['user_id', 'item_id'], how='left', validate='one_to_one')
            name = f'affinity_{relation}_{suffix}_sum'
            denominator = 'user_count_all' if days is None else f'user_count_{days}d'
            frame[name + '_share'] = frame[name] / frame[denominator].clip(lower=1)
        last = expanded.groupby(['user_id', 'token']).date.max().rename('last').reset_index()
        recency = pair_tokens.merge(last, on=['user_id', 'token'], how='left')
        recency['days'] = (cutoff - recency['last']).dt.days
        recency = recency.groupby(['user_id', 'item_id']).days.min().rename(f'affinity_{relation}_last_days')
        frame = frame.merge(recency, on=['user_id', 'item_id'], how='left')
    for column in CATEGORY_COLUMNS:
        vocabulary = users[column] if column in users else metadata[column]
        dtype = pd.CategoricalDtype(sorted(set(vocabulary) | {'unknown'}))
        frame[column] = frame[column].fillna('unknown').astype(str).astype(dtype)
    for column in frame.columns.difference(CATEGORY_COLUMNS + ['user_id', 'item_id']):
        frame[column] = pd.to_numeric(frame[column], errors='raise').fillna(-1).astype('float32')
    return frame


def targets(events, user_ids, cutoff, horizon):
    # Foreign users cannot affect these labels; avoid a Python set of millions of pairs.
    events = events[events.user_id.isin(user_ids)]
    future = events[(events.date >= cutoff) & (events.date < cutoff + pd.Timedelta(days=horizon))]
    seen = set(map(tuple, events[events.date < cutoff][['user_id', 'item_id']].itertuples(index=False, name=None)))
    result = {}
    for user, item in future[['user_id', 'item_id']].itertuples(index=False, name=None):
        if (user, item) not in seen:
            result.setdefault(user, set()).add(item)
    return result


def evaluate(frame, scores, truth, k=10):
    """Macro metrics over ALL eligible users with positives, including retrieval misses."""
    ranked = frame[['user_id', 'item_id']].assign(score=np.asarray(scores))
    ranked = ranked.sort_values(['user_id', 'score', 'item_id'], ascending=[True, False, True])
    recommendations = ranked.groupby('user_id').head(k).groupby('user_id').item_id.agg(list).to_dict()
    candidate_sets = frame.groupby('user_id').item_id.agg(set).to_dict()
    rows = []
    for user, relevant in sorted(truth.items()):
        hits = np.array([int(i in relevant) for i in recommendations.get(user, [])])
        ideal = np.sum(1 / np.log2(np.arange(min(k, len(relevant))) + 2))
        dcg = np.sum(hits / np.log2(np.arange(len(hits)) + 2))
        ap = np.sum(hits * np.cumsum(hits) / (np.arange(len(hits)) + 1)) / min(k, len(relevant))
        rows.append((user, dcg / ideal, hits.sum() / len(relevant), ap,
                     len(relevant & candidate_sets.get(user, set())) / len(relevant)))
    per_user = pd.DataFrame(rows, columns=['user_id', 'ndcg', 'recall', 'map', 'candidate_recall'])
    if per_user.empty:
        raise ValueError('No evaluable users in time interval')
    return {**{f'{c}@{k}' if c != 'candidate_recall' else c: float(per_user[c].mean())
               for c in per_user.columns[1:]}, 'users': len(per_user)}, per_user


def prepare(args, config):
    start = time.monotonic()
    events, users, items = read_data(args.data_dir)
    rng = np.random.default_rng(config['seed'])
    population = np.sort(events.user_id.unique())
    chosen = np.sort(rng.choice(population, min(config['sample_users'], len(population)), replace=False))
    pd.Series(chosen, name='user_id').to_csv(args.output / 'sample_users.csv', index=False)
    end = events.date.max() + pd.Timedelta(days=1)
    horizon = config['horizon_days']
    cuts = [end - pd.Timedelta(days=horizon * i) for i in range(config['train_weeks'] + 2, 0, -1)]
    manifest = {'config': config, 'source': str(args.data_dir / 'data_original.zip'),
                'events': len(events), 'min_date': str(events.date.min()), 'max_date': str(events.date.max()),
                'splits': [], 'target': 'new observed user-item pair in next seven days'}
    for index, cutoff in enumerate(cuts):
        split = 'train' if index < config['train_weeks'] else ('validation' if index == config['train_weeks'] else 'test')
        log(f'Preparing {split} at {cutoff.date()}')
        history = history_before(events, cutoff)
        # Eligibility uses history only; evaluation selects positives afterwards.
        eligible = np.intersect1d(chosen, history.user_id.unique())
        candidate = make_candidates(history, eligible, items, config, cutoff)
        truth = targets(events, eligible, cutoff, horizon)
        candidate['label'] = [int(i in truth.get(u, set())) for u, i in candidate[['user_id', 'item_id']].itertuples(index=False, name=None)]
        # No-positive training queries carry no pairwise ranking signal.
        if split == 'train':
            candidate = candidate[candidate.groupby('user_id').label.transform('sum') > 0]
        batches = []
        batch_users = candidate.user_id.unique()
        for offset in range(0, len(batch_users), 2000):
            batch = candidate[candidate.user_id.isin(batch_users[offset:offset + 2000])]
            batches.append(make_features(batch, history, users, items, cutoff, config['windows']))
            if len(batch_users) > 2000:
                log(f'  Features: {min(offset + 2000, len(batch_users))}/{len(batch_users)} users')
        if not batches:
            raise ValueError(f'No usable candidate queries at {cutoff}')
        features = pd.concat(batches, ignore_index=True)
        features['query_id'] = index * (int(events.user_id.max()) + 1) + features.user_id
        path = args.output / f'{split}_{cutoff.date()}.pkl'
        features.to_pickle(path)
        truth_path = path.with_suffix('.truth.json')
        truth_path.write_text(json.dumps({str(k): sorted(v) for k, v in truth.items()}))
        record = {'split': split, 'cutoff': str(cutoff.date()), 'label_end_exclusive': str((cutoff + pd.Timedelta(days=horizon)).date()),
                  'path': path.name, 'truth': truth_path.name, 'rows': len(features),
                  'positive_rows': int(features.label.sum()), 'eligible_users': len(eligible), 'positive_users': len(truth)}
        manifest['splits'].append(record)
        log(str(record))
    manifest['prepare_seconds'] = time.monotonic() - start
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))


def load_split(output, manifest, name):
    records = [r for r in manifest['splits'] if r['split'] == name]
    frame = pd.concat([pd.read_pickle(output / r['path']) for r in records], ignore_index=True)
    frame = frame.sort_values(['query_id', 'item_id']).reset_index(drop=True)
    truth = {}
    if name != 'train':
        raw = json.loads((output / records[0]['truth']).read_text())
        truth = {int(k): set(v) for k, v in raw.items()}
    return frame, truth


def train(args, config):
    try:
        from catboost import CatBoostRanker, Pool
    except ImportError as error:
        raise SystemExit('CatBoost unavailable. Install baseline/requirements.txt in Python 3.12 or 3.13.') from error
    manifest = json.loads((args.output / 'manifest.json').read_text())
    if config != manifest['config']:
        raise ValueError('Configuration differs from prepared data; run prepare again')
    training, _ = load_split(args.output, manifest, 'train')
    validation, truth = load_split(args.output, manifest, 'validation')
    if training.label.nunique() != 2 or validation.label.nunique() != 2:
        raise ValueError('Train and validation must contain positive and negative candidates')
    columns = [c for c in training if c not in ['user_id', 'item_id', 'label', 'query_id']]
    def pool(frame):
        return Pool(frame[columns], label=frame.label, group_id=frame.query_id,
                    cat_features=CATEGORY_COLUMNS)
    log(f'Training {len(training)} rows, {len(columns)} features')
    started = time.monotonic()
    model = CatBoostRanker(iterations=config['iterations'], depth=config['depth'],
                          learning_rate=config['learning_rate'], loss_function='YetiRank',
                          eval_metric='NDCG:top=10', random_seed=config['seed'],
                          thread_count=config['thread_count'], allow_writing_files=False)
    model.fit(pool(training), eval_set=pool(validation), early_stopping_rounds=80, verbose=50)
    model.save_model(str(args.output / 'model.cbm'))
    report = {'train_seconds': time.monotonic() - started, 'best_iteration': model.best_iteration_,
              'features': columns, 'validation': {}, 'test': {}}
    score_model(args, model, columns, manifest, report, 'validation')
    # PredictionValuesChange works for ranking without materializing a training Pool.
    importance = model.get_feature_importance(type='PredictionValuesChange')
    pd.DataFrame({'feature': columns, 'importance': importance}).sort_values('importance', ascending=False).to_csv(
        args.output / 'feature_importance.csv', index=False)
    report['peak_process_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    report['prepare_seconds'] = manifest['prepare_seconds']
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2))


def score_model(args, model, columns, manifest, report, split):
    frame, truth = load_split(args.output, manifest, split)
    report[split] = {}
    predict_start = time.monotonic()
    prediction = model.predict(frame[columns])
    report[split]['prediction_seconds'] = time.monotonic() - predict_start
    individual = {}
    for name, scores in [('popularity', frame.item_count_30d), ('catboost', prediction)]:
        metrics, per_user = evaluate(frame, scores, truth)
        individual[name] = per_user.set_index('user_id')
        report[split][name] = metrics
        per_user.to_csv(args.output / f'{split}_{name}_per_user.csv', index=False)
        log(f'{split} {name}: {metrics}')
    difference = (individual['catboost'].ndcg - individual['popularity'].ndcg).to_numpy()
    rng = np.random.default_rng(manifest['config']['seed'])
    bootstrap = [float(rng.choice(difference, len(difference), replace=True).mean()) for _ in range(1000)]
    report[split]['ndcg_delta'] = float(difference.mean())
    report[split]['ndcg_delta_95pct_ci'] = np.quantile(bootstrap, [0.025, 0.975]).tolist()
    scored = frame[['user_id', 'item_id']].assign(score=prediction)
    scored.sort_values(['user_id', 'score', 'item_id'], ascending=[True, False, True]).groupby('user_id').head(10).to_csv(
        args.output / f'{split}_recommendations.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'train', 'run', 'evaluate-popularity', 'evaluate-test'])
    parser.add_argument('--config', type=Path, default=ROOT / 'baseline/configs/default.json')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'KION_DATASET')
    parser.add_argument('--output', type=Path, default=ROOT / 'baseline/artifacts/default')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if not {7, 30}.issubset(config['windows']):
        parser.error('windows must contain 7 and 30')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command in ['prepare', 'run']:
        prepare(args, config)
    if args.command in ['train', 'run']:
        train(args, config)
    if args.command == 'evaluate-test':
        from catboost import CatBoostRanker
        manifest = json.loads((args.output / 'manifest.json').read_text())
        report = json.loads((args.output / 'metrics.json').read_text())
        if config != manifest['config']:
            raise ValueError('Configuration differs from prepared data')
        model = CatBoostRanker()
        model.load_model(str(args.output / 'model.cbm'))
        score_model(args, model, report['features'], manifest, report, 'test')
        (args.output / 'metrics.json').write_text(json.dumps(report, indent=2))
    if args.command == 'evaluate-popularity':
        manifest = json.loads((args.output / 'manifest.json').read_text())
        report = {}
        for split in ['validation', 'test']:
            frame, truth = load_split(args.output, manifest, split)
            report[split], _ = evaluate(frame, frame.item_count_30d, truth)
        (args.output / 'popularity_metrics.json').write_text(json.dumps(report, indent=2))
        log(str(report))


if __name__ == '__main__':
    main()
