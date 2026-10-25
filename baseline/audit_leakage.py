"""Audit real cached graphs and future invariance without training or test labels."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from baseline.aggregates import ManualAggregates
from baseline.pipeline import ROOT, history_before, log, make_candidates, make_features, read_data


def main():
    started = time.monotonic()
    base = ROOT/'baseline/artifacts/default'
    strong = ROOT/'baseline/artifacts/strong'
    config = json.loads((ROOT/'baseline/configs/strong.json').read_text())
    config['extended_aggregates'] = True
    manifest = json.loads((base/'manifest.json').read_text())
    signature = hashlib.sha256(json.dumps({k: config[k] for k in [
        'neighbors','minimum_support','similarity_shrinkage','graph_history_limit']}, sort_keys=True).encode()).hexdigest()[:10]
    events, users, items = read_data(ROOT/'KION_DATASET')
    selected = pd.read_csv(base/'sample_users.csv').user_id.to_numpy()[:100]
    records = []
    for split in manifest['splits']:
        if split['split'] == 'test':
            continue
        cutoff = pd.Timestamp(split['cutoff'])
        history = history_before(events, cutoff)
        eligible = np.intersect1d(selected, history.user_id.unique())
        # No graph cache is passed: reconstruct from the complete strict past.
        ctx = ManualAggregates(history, users, items, cutoff, config, user_ids=eligible)
        cached = sparse.load_npz(strong/f'graph_{cutoff.date()}_{signature}.npz').tocsr()
        difference = (cached-ctx.graph).tocsr()
        difference.eliminate_zeros()
        if difference.nnz:
            raise AssertionError(f'Graph differs from past-only rebuild at {cutoff}')
        record = {'cutoff': str(cutoff.date()), 'history_rows': len(history),
                  'max_history_date': str(history.date.max()), 'cached_graph_matches_past_rebuild': True}
        if split['split'] == 'validation':
            original = make_candidates(history, eligible, items, manifest['config'], cutoff)
            candidates = ctx.expand_candidates(original)
            before = pd.concat([make_features(candidates, history, users, items, cutoff,
                                               manifest['config']['windows']), ctx.features(candidates)], axis=1)
            # Replace every future pair by different items/users, keeping all past intact.
            future = events[events.date >= cutoff].copy()
            future['item_id'] = np.roll(future.item_id.to_numpy(), 1)
            future['user_id'] = np.roll(future.user_id.to_numpy(), 13)
            changed = history_before(pd.concat([history, future], ignore_index=True), cutoff)
            pd.testing.assert_frame_equal(history.reset_index(drop=True), changed.reset_index(drop=True))
            other = ManualAggregates(changed, users, items, cutoff, config, user_ids=eligible)
            after_candidates = other.expand_candidates(make_candidates(changed, eligible, items, manifest['config'], cutoff))
            pd.testing.assert_frame_equal(candidates, after_candidates)
            after = pd.concat([make_features(after_candidates, changed, users, items, cutoff,
                                              manifest['config']['windows']), other.features(after_candidates)], axis=1)
            pd.testing.assert_frame_equal(before, after)
            # A different set of evaluated users cannot change a user's candidates/features.
            one = int(eligible[0])
            only = ManualAggregates(history, users, items, cutoff, config,
                                   graph_path=strong/f'graph_{cutoff.date()}_{signature}.npz', user_ids=[one])
            one_candidates = only.expand_candidates(make_candidates(history, [one], items, manifest['config'], cutoff))
            pd.testing.assert_frame_equal(candidates[candidates.user_id==one].reset_index(drop=True), one_candidates)
            pd.testing.assert_frame_equal(ctx.features(candidates[candidates.user_id==one].reset_index(drop=True)),
                                          only.features(one_candidates))
            record.update(future_rows_replaced=len(future), users_checked=len(eligible),
                          candidate_rows=len(candidates), features_checked=len(before.columns)-2,
                          future_invariance=True, evaluation_user_filter_invariance=True)
        records.append(record)
        log(record)
        del ctx, cached, difference
    report = {'checks': records, 'elapsed_seconds': time.monotonic()-started,
              'scope': 'dynamic aggregates and graph caches; metadata availability cannot be proved from snapshots',
              'source_sha256': {name: hashlib.sha256((ROOT/'baseline'/name).read_bytes()).hexdigest()
                                for name in ['pipeline.py','aggregates.py','strong.py']}}
    destination = ROOT/'baseline/artifacts/leakage_audit.json'
    destination.write_text(json.dumps(report, indent=2))
    log(f'Audit passed: {destination}')


if __name__ == '__main__':
    main()
