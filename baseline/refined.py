"""Past-only candidate provenance, local-interest and preference-lift features."""
from collections import Counter

import numpy as np
import pandas as pd


class RefinedFeatures:
    def __init__(self, context, original):
        self.ctx = context
        self.original = set(original[['user_id', 'item_id']].itertuples(index=False, name=None))
        self.fresh = self._ranks(context.item_counts[3])
        counts, _ = context.cohorts['age', 30]
        self.age_ranks = {age: self._ranks(group.droplevel(0))
                          for age, group in counts.groupby(level=0)}
        self.priors = {}
        # Same token mass convention for personal profiles and global priors.
        frequency = context.history.groupby('item_id').size()
        history_indices = context.item_index.get_indexer(frequency.index)
        if np.any(history_indices < 0):
            raise ValueError('History contains unknown item')
        for relation, tokens in context.tokens.items():
            total = Counter()
            for index, count in zip(history_indices, frequency.to_numpy()):
                for token in tokens[index]:
                    total[token] += count / len(tokens[index])
            mass = sum(total.values())
            self.priors[relation] = {token: count / max(mass, 1) for token, count in total.items()}

    @staticmethod
    def _ranks(counts):
        table = counts.rename('score').rename_axis('item_id').reset_index()
        table = table.sort_values(['score', 'item_id'], ascending=[False, True])
        return {int(item): rank for rank, item in enumerate(table.item_id, 1)}

    def features(self, candidates):
        ctx = self.ctx
        source_names = ['original', 'co', 'fresh', 'age', 'count', 'co_rank',
                        'co_score', 'fresh_rank', 'age_rank']
        interest_names = ['top3_mean', 'top1_fraction', 'weighted_age', 'recent7_mean',
                          'older_mean', 'recent_minus_older', 'last_day_mean']
        lift_names = [f'{relation}_{kind}' for relation in ctx.tokens
                      for kind in ['mean', 'max', 'support']]
        sources = np.zeros((len(candidates), len(source_names)), dtype='float32')
        interests = np.zeros((len(candidates), len(interest_names)), dtype='float32')
        lifts = np.zeros((len(candidates), len(lift_names)), dtype='float32')
        sources[:, [5, 7, 8]] = 99999
        interests[:, 2] = 999
        for user, positions in candidates.groupby('user_id', sort=False).indices.items():
            item_ids = candidates.iloc[positions].item_id.to_numpy()
            indices = ctx.item_index.get_indexer(item_ids)
            if np.any(indices < 0):
                raise ValueError('Unknown candidate item')
            seeds, ages = ctx.histories.get(int(user), (np.array([], dtype=int), np.array([], dtype=int)))
            neighbors, scores = ctx.neighbor_scores(user)
            seen = set(ctx.item_ids[seeds])
            available = [(int(ctx.item_ids[i]), float(s)) for i, s in zip(neighbors, scores)
                         if ctx.item_ids[i] not in seen]
            co = {item: (rank, score) for rank, (item, score) in enumerate(available, 1)}
            age = ctx.user_groups['age'].get(user, 'unknown')
            age_ranks = self.age_ranks.get(age, {})
            for position, item in zip(positions, item_ids):
                co_rank, score = co.get(int(item), (99999, 0))
                fresh_rank = self.fresh.get(int(item), 99999)
                age_rank = age_ranks.get(int(item), 99999)
                flags = [(user, item) in self.original, co_rank <= ctx.config['co_candidates'],
                         fresh_rank <= ctx.config['fresh_candidates'], age_rank <= ctx.config['cohort_candidates']]
                sources[position] = [*flags, sum(flags), co_rank, score, fresh_rank, age_rank]
            if not len(seeds):
                continue
            ages = np.asarray(ages)
            links = ctx.graph[seeds][:, indices].toarray()
            mass = links.sum(axis=0)
            top = np.sort(links, axis=0)[-min(3, len(seeds)):]
            recent, older = ages <= 7, ages > 7
            recent_mean = links[recent].mean(axis=0) if recent.any() else np.zeros(len(indices))
            older_mean = links[older].mean(axis=0) if older.any() else np.zeros(len(indices))
            # Cutoff at midnight: ages 1..7 are the previous seven complete dates.
            last_day = links[ages == ages.min()].mean(axis=0)
            weighted_age = np.divide((links * ages[:, None]).sum(axis=0), mass,
                                     out=np.full(len(indices), 999.), where=mass > 0)
            top_fraction = np.divide(links.max(axis=0), mass,
                                     out=np.zeros(len(indices)), where=mass > 0)
            interests[positions] = np.column_stack([top.mean(axis=0), top_fraction, weighted_age,
                                                    recent_mean, older_mean, recent_mean-older_mean, last_day])
            for number, (relation, tokens) in enumerate(ctx.tokens.items()):
                profile = Counter()
                for seed in seeds:
                    for token in tokens[seed]:
                        profile[token] += 1 / len(tokens[seed])
                profile_mass = sum(profile.values())
                prior = self.priors[relation]
                for position, item in zip(positions, indices):
                    candidate_tokens = tokens[item]
                    if not candidate_tokens:
                        continue
                    # Bayesian shrinkage of a token distribution towards its global prior.
                    ratios = [(profile[token] + 10 * prior.get(token, 0)) /
                              ((profile_mass + 10) * prior[token])
                              if prior.get(token, 0) > 0 else 1. for token in candidate_tokens]
                    # Bounded log-lift limits the influence of very rare metadata.
                    transformed = np.log(np.clip(ratios, .01, 100))
                    lifts[position, number*3:number*3+3] = [transformed.mean(), transformed.max(),
                                                           sum(profile[t] for t in candidate_tokens)]
        return pd.concat([
            pd.DataFrame(sources, columns=['ref_source_'+n for n in source_names], index=candidates.index),
            pd.DataFrame(interests, columns=['ref_interest_'+n for n in interest_names], index=candidates.index),
            pd.DataFrame(lifts, columns=['ref_lift_'+n for n in lift_names], index=candidates.index),
        ], axis=1)
