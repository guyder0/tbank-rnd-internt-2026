"""Strict-time KION queries and local synthetic smoke fixtures."""
from itertools import islice

import numpy as np
import pandas as pd

from baseline.pipeline import history_before, targets


def prepare_queries(events, items, chosen, cutoffs, top_k):
    first = pd.Timestamp(cutoffs[0])
    catalog_ids = np.sort(events.loc[events.date < first, 'item_id'].unique())
    catalog = items.set_index('item_id').loc[catalog_ids].reset_index()
    known = set(map(int, catalog_ids))
    splits = []
    for number, date in enumerate(cutoffs):
        cutoff = pd.Timestamp(date)
        history = history_before(events, cutoff)
        eligible = np.intersect1d(chosen, history.user_id.unique())
        truth = targets(events, eligible, cutoff, 7)
        local = history[history.user_id.isin(truth)].sort_values(['user_id', 'date', 'item_id'])
        recent = history[history.date >= cutoff-pd.Timedelta(days=30)]
        counts = recent.groupby('item_id').size().rename('n').reset_index()
        counts = counts.sort_values(['n', 'item_id'], ascending=[False, True])
        order = counts.item_id.tolist()
        catalog_order = [int(i) for i in order if i in known]
        queries = []
        for user, part in local.groupby('user_id', sort=True):
            seen = set(map(int, part.item_id))
            queries.append({'user_id': int(user), 'items': part.item_id.to_numpy(dtype='int64'),
                            'ages': (cutoff-part.date).dt.days.to_numpy(dtype='int64'),
                            'positives': sorted(truth[int(user)]),
                            'popularity': list(islice((int(i) for i in order if i not in seen), top_k)),
                            'popularity_catalog': list(islice((i for i in catalog_order if i not in seen), top_k))})
        if not queries:
            raise ValueError(f'No positive warm queries at {cutoff}')
        splits.append({'name': 'validation' if number == len(cutoffs)-1 else 'train',
                       'cutoff': str(cutoff.date()), 'max_history_date': str(history.date.max()),
                       'queries': queries})
    return catalog, splits


def cap_history(query, limit):
    """Keep recent complete date groups; never truncate a group by item ID."""
    ages = query['ages']
    if len(ages) <= limit:
        return query['items'], ages
    boundary = np.sort(ages)[limit-1]
    keep = ages <= boundary
    return query['items'][keep], ages[keep]


def synthetic_data():
    """No downloads: repeated users with shared genre preferences and full past."""
    ids = np.arange(100, 132)
    items = pd.DataFrame({'item_id': ids, 'title': [f'Фильм номер {i}' for i in ids],
                          'genres': ['драма' if i % 2 else 'комедия' for i in ids],
                          'countries': ['Россия']*len(ids),
                          'directors': [f'Автор {i%4}' for i in ids]})
    rows = [(900, int(item), pd.Timestamp('2021-06-01')) for item in ids]
    dates = ['2021-07-10','2021-07-20','2021-07-27','2021-08-03','2021-08-10']
    for user in range(1, 25):
        for number, date in enumerate(dates):
            item = int(ids[(user%4)*8+number])
            rows.append((user, item, pd.Timestamp(date)))
    return pd.DataFrame(rows, columns=['user_id','item_id','date']), items, np.arange(1,25)
