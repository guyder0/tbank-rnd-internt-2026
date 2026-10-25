"""Batch feature construction using the established baseline definitions."""
from copy import copy

import numpy as np
import pandas as pd

from baseline.aggregates import ManualAggregates
from baseline.pipeline import make_features, token_table
from baseline.refined import RefinedFeatures


class CandidateFactory:
    """Precompute global retrieval statistics once, preserving baseline candidates."""
    def __init__(self, history, query_users, items, cutoff, config):
        self.config = config
        recent = history[history.date >= cutoff-pd.Timedelta(days=30)]
        counts = recent.groupby('item_id').size().sort_values(ascending=False)
        fallback = history.groupby('item_id').size().sort_values(ascending=False)
        self.global_items = list(dict.fromkeys([*counts.index, *fallback.index]))[:config['popular_candidates']]
        tokens = token_table(items, 'genres')
        preferences = history[history.user_id.isin(query_users)].merge(tokens, on='item_id')
        preferences = preferences.groupby(['user_id','token']).size().rename('n').reset_index()
        preferences = preferences.sort_values(['n','token'], ascending=[False, True])
        self.top_genres = preferences.groupby('user_id').head(3).groupby('user_id').token.agg(list).to_dict()
        popularity = recent.merge(tokens, on='item_id').groupby(['token','item_id']).size()
        gp = popularity.rename('n').reset_index().sort_values(['n','item_id'], ascending=[False, True])
        self.genre_items = gp.groupby('token').item_id.agg(list).to_dict()
        self.seen = history[history.user_id.isin(query_users)].groupby('user_id').item_id.agg(set).to_dict()

    def build(self, ids):
        rows = []
        for user in sorted(ids):
            personalized = []
            for genre in self.top_genres.get(user, []):
                personalized.extend(self.genre_items.get(genre, [])[:self.config['genre_candidates']//3])
            rows.extend((int(user), int(item)) for item in
                        sorted(set(self.global_items+personalized)-self.seen.get(user, set())))
        return pd.DataFrame(rows, columns=['user_id','item_id']).astype({'user_id':'int64','item_id':'int64'})


class FeatureBuilder:
    def __init__(self, history, users, items, cutoff, ids, config):
        if history.empty:
            raise ValueError(f'No history before {cutoff.date()}')
        self.history, self.users, self.items = history, users, items
        self.cutoff, self.config = cutoff, config
        self.factory = CandidateFactory(history, ids, items, cutoff, config)
        self.ctx = ManualAggregates(history, users, items, cutoff, config, user_ids=ids)
        self.refined = RefinedFeatures(self.ctx, pd.DataFrame(columns=['user_id','item_id']))

    def build(self, ids):
        original = self.factory.build(ids)
        local = copy(self.ctx)
        local.histories = {int(u): self.ctx.histories[int(u)] for u in ids if int(u) in self.ctx.histories}
        candidates = local.expand_candidates(original)
        # No personal history: retain original popularity/metadata retrieval.
        cold = original.loc[~original.user_id.isin(local.histories)]
        candidates = pd.concat([candidates, cold], ignore_index=True).drop_duplicates()
        candidates = candidates.sort_values(['user_id','item_id']).reset_index(drop=True)
        if candidates.empty:
            return pd.DataFrame(columns=['user_id','item_id',*self.config['feature_columns']])
        self.refined.original = set(original.itertuples(index=False, name=None))
        frame = pd.concat([
            make_features(candidates, self.history, self.users, self.items, self.cutoff, self.config['windows']),
            self.ctx.features(candidates), self.refined.features(candidates),
        ], axis=1)
        frame = frame[['user_id','item_id',*self.config['feature_columns']]]
        if frame.columns.duplicated().any() or frame.isna().any().any():
            raise ValueError('Duplicate or missing feature values')
        numeric = frame.select_dtypes(include='number')
        if not np.isfinite(numeric.to_numpy()).all():
            raise ValueError('Nonfinite feature values')
        return frame
