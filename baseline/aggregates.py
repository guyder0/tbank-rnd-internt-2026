"""Fixed manual aggregate definitions. No fitted embeddings or latent factors."""
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from baseline.pipeline import log


class ManualAggregates:
    def __init__(self, history, users, items, cutoff, config, graph_path=None, user_ids=None):
        if len(history) and history.date.max() >= cutoff:
            raise ValueError('Aggregate context includes future events')
        self.history = history
        self.cutoff = cutoff
        self.config = config
        self.items = items.sort_values('item_id').reset_index(drop=True)
        self.item_ids = self.items.item_id.to_numpy()
        self.item_index = pd.Index(self.item_ids)
        self.users = users.set_index('user_id')
        self.user_groups = {}
        for name in ['age', 'sex', 'kids_flg', 'income']:
            self.user_groups[name] = self.users[name].astype('string').fillna('unknown')
        self.user_groups['age_sex'] = self.user_groups['age'] + '|' + self.user_groups['sex']
        self.item_counts = {}
        for days in [1, 3, 7, 14, 30]:
            recent = history[history.date >= cutoff - pd.Timedelta(days=days)]
            self.item_counts[days] = recent.groupby('item_id').size()
        self.cohorts = {}
        recent = history[history.date >= cutoff - pd.Timedelta(days=30)]
        global_counts = self.item_counts[30]
        self.global_prior = global_counts / max(1, len(recent))
        for group, mapping in self.user_groups.items():
            grouped = recent[['user_id', 'item_id']].copy()
            grouped['cohort'] = grouped.user_id.map(mapping).fillna('unknown')
            for days in [7, 30]:
                selected = grouped if days == 30 else grouped.loc[recent.date >= cutoff - pd.Timedelta(days=7)]
                self.cohorts[group, days] = (
                    selected.groupby(['cohort', 'item_id']).size(), selected.groupby('cohort').size())
        ordered = history.sort_values(['user_id', 'date', 'item_id'], ascending=[True, False, True])
        # Keep all history for user features. The explicit graph-only cap controls heavy-user cost.
        local = ordered if user_ids is None else ordered[ordered.user_id.isin(user_ids)]
        self.histories = {int(user): (self.item_index.get_indexer(part.item_id),
                                     (cutoff - part.date).dt.days.to_numpy())
                          for user, part in local.groupby('user_id', sort=False)}
        if graph_path is not None and Path(graph_path).exists():
            self.graph = sparse.load_npz(graph_path).tocsr()
        else:
            self.graph = self.build_graph(ordered)
            if graph_path is not None:
                sparse.save_npz(graph_path, self.graph)
        self.tokens = {}
        relations = ['actors', 'studios', 'content_type', 'age_rating']
        if config.get('extended_aggregates'):
            relations += ['genres', 'countries', 'directors']
        for relation in relations:
            if relation not in self.items:
                continue
            values = self.items[relation].astype('string').fillna('')
            self.tokens[relation] = [tuple(dict.fromkeys(t.strip() for t in value.split(',') if t.strip()))
                                     for value in values]

    def build_graph(self, ordered):
        bounded = ordered.groupby('user_id', sort=False).head(self.config['graph_history_limit'])
        rows, _ = pd.factorize(bounded.user_id, sort=True)
        cols = self.item_index.get_indexer(bounded.item_id)
        matrix = sparse.csr_matrix((np.ones(len(rows), dtype='float32'), (rows, cols)),
                                   shape=(int(rows.max()) + 1, len(self.item_ids)))
        frequency = np.asarray(matrix.sum(axis=0)).ravel()
        all_rows, all_cols, all_values = [], [], []
        transpose = matrix.T.tocsr()
        for start in range(0, len(self.item_ids), 256):
            stop = min(start + 256, len(self.item_ids))
            co = (transpose[start:stop] @ matrix).toarray()
            norm = np.sqrt(frequency[start:stop, None] * frequency[None, :])
            values = np.divide(co, norm, out=np.zeros_like(co), where=norm > 0)
            values *= co / (co + self.config['similarity_shrinkage'])
            values[co < self.config['minimum_support']] = 0
            values[np.arange(stop-start), np.arange(start, stop)] = 0
            for offset, row in enumerate(values):
                count = min(self.config['neighbors'], len(row))
                indices = np.argpartition(row, -count)[-count:]
                indices = indices[row[indices] > 0]
                indices = indices[np.lexsort((indices, -row[indices]))]
                all_rows.extend([start + offset] * len(indices))
                all_cols.extend(indices.tolist())
                all_values.extend(row[indices].tolist())
            if start % 2048 == 0:
                log(f'  Co-view graph: {stop}/{len(self.item_ids)} items')
        result = sparse.csr_matrix((np.array(all_values, dtype='float32'), (all_rows, all_cols)),
                                  shape=(len(self.item_ids), len(self.item_ids)))
        result.sort_indices()
        return result

    def neighbor_scores(self, user):
        seeds, ages = self.histories.get(int(user), ([], []))
        columns, values = [], []
        for seed, age in zip(seeds, ages):
            start, stop = self.graph.indptr[seed:seed+2]
            columns.extend(self.graph.indices[start:stop])
            values.extend(self.graph.data[start:stop] * (0.25 + 0.75 * np.exp(-age / 30)))
        if not columns:
            return np.array([], dtype='int64'), np.array([], dtype='float32')
        unique, inverse = np.unique(columns, return_inverse=True)
        scores = np.bincount(inverse, weights=values)
        order = np.lexsort((unique, -scores))
        return unique[order], scores[order]

    def expand_candidates(self, original):
        rows = []
        fresh = self.item_counts[3].sort_values(ascending=False, kind='stable').index[:self.config['fresh_candidates']]
        age_counts, _ = self.cohorts['age', 30]
        age_candidates = {}
        for age, group in age_counts.groupby(level=0):
            age_candidates[age] = group.droplevel(0).sort_values(ascending=False, kind='stable').index[:self.config['cohort_candidates']]
        original_items = original.groupby('user_id').item_id.agg(list).to_dict()
        for user in sorted(self.histories):
            neighbors, _ = self.neighbor_scores(user)
            seeds, _ = self.histories.get(int(user), ([], []))
            seen = set(self.item_ids[seeds])
            co_items = [item for item in self.item_ids[neighbors] if item not in seen][:self.config['co_candidates']]
            age = self.user_groups['age'].get(user, 'unknown')
            candidates = set(original_items.get(user, [])) | set(co_items) | set(fresh) | set(age_candidates.get(age, []))
            rows.extend((user, int(item)) for item in sorted(candidates - seen))
        return pd.DataFrame(rows, columns=['user_id', 'item_id'])

    def features(self, candidates):
        result = pd.DataFrame(index=candidates.index)
        for days in [1, 3, 14]:
            result[f'manual_item_count_{days}d'] = candidates.item_id.map(self.item_counts[days]).fillna(0).to_numpy()
        recent7 = candidates.item_id.map(self.item_counts[7]).fillna(0).to_numpy()
        recent14 = candidates.item_id.map(self.item_counts[14]).fillna(0).to_numpy()
        recent3 = result.manual_item_count_3d.to_numpy()
        result['manual_item_week_acceleration'] = (recent7 + 1) / (recent14 - recent7 + 1)
        result['manual_item_short_acceleration'] = (recent3 + 1) / ((recent7 - recent3) * .75 + 1)
        prior = candidates.item_id.map(self.global_prior).fillna(0).to_numpy()
        for group, mapping in self.user_groups.items():
            cohort = candidates.user_id.map(mapping).fillna('unknown')
            index = pd.MultiIndex.from_arrays([cohort, candidates.item_id])
            for days in [7, 30]:
                counts, totals = self.cohorts[group, days]
                hits = counts.reindex(index, fill_value=0).to_numpy(dtype='float32')
                total = cohort.map(totals).fillna(0).to_numpy(dtype='float32')
                probability = (hits + 100 * prior) / (total + 100)
                prefix = f'manual_cohort_{group}_{days}d'
                result[prefix + '_count'] = hits
                result[prefix + '_probability'] = probability
                result[prefix + '_lift'] = probability / np.maximum(prior, 1e-7)
        graph_names = ['sum', 'max', 'support', 'last1', 'last3', 'last5', 'decay7', 'decay30', 'mean']
        graph_features = np.zeros((len(candidates), len(graph_names)), dtype='float32')
        temporal_names = ['history_size', 'active_days', 'mean_gap', 'std_gap', 'last_gap', 'count_1d', 'count_3d', 'decay7', 'decay30']
        temporal = np.zeros((len(candidates), len(temporal_names)), dtype='float32')
        metadata_names = [f'manual_{relation}_{kind}' for relation in self.tokens
                          for kind in ['sum', 'max', 'share', 'recent7_share', 'decay30_share', 'last1_overlap', 'last3_overlap']]
        metadata = np.zeros((len(candidates), len(metadata_names)), dtype='float32')
        for user, positions in candidates.groupby('user_id', sort=False).indices.items():
            candidate_items = self.item_index.get_indexer(candidates.iloc[positions].item_id)
            if np.any(candidate_items < 0):
                raise ValueError('Unknown candidate item')
            seeds, ages = self.histories.get(int(user), (np.array([], dtype=int), np.array([], dtype=int)))
            if not len(seeds):
                continue
            order = np.argsort(candidate_items)
            sorted_items = candidate_items[order]
            for number, (seed, age) in enumerate(zip(seeds, ages)):
                start, stop = self.graph.indptr[seed:seed+2]
                neighbors = self.graph.indices[start:stop]
                scores = self.graph.data[start:stop]
                where = np.searchsorted(sorted_items, neighbors)
                valid = where < len(sorted_items)
                valid[valid] &= sorted_items[where[valid]] == neighbors[valid]
                targets = positions[order[where[valid]]]
                score = scores[valid]
                graph_features[targets, 0] += score
                graph_features[targets, 1] = np.maximum(graph_features[targets, 1], score)
                graph_features[targets, 2] += 1
                if number == 0: graph_features[targets, 3] += score
                if number < 3: graph_features[targets, 4] += score
                if number < 5: graph_features[targets, 5] += score
                graph_features[targets, 6] += score * np.exp(-age/7)
                graph_features[targets, 7] += score * np.exp(-age/30)
            graph_features[positions, 8] = graph_features[positions, 0] / len(seeds)
            days = np.unique(ages)
            gaps = np.diff(days)
            decay_weights = np.exp(-ages/30)
            recent_count = max(1, np.sum(ages <= 7))
            decay_mass = max(1e-6, decay_weights.sum())
            temporal[positions] = [len(seeds), len(days), np.mean(gaps) if len(gaps) else 0,
                                   np.std(gaps) if len(gaps) else 0, gaps[0] if len(gaps) else 0,
                                   np.sum(ages <= 1), np.sum(ages <= 3), np.exp(-ages/7).sum(), np.exp(-ages/30).sum()]
            for relation_index, (relation, tokens) in enumerate(self.tokens.items()):
                total, recent, decayed = Counter(), Counter(), Counter()
                for seed, age, weight in zip(seeds, ages, decay_weights):
                    for token in tokens[seed]:
                        total[token] += 1
                        if age <= 7: recent[token] += 1
                        decayed[token] += weight
                last1 = set(tokens[seeds[0]])
                last3 = set(token for seed in seeds[:3] for token in tokens[seed])
                for position, item in zip(positions, candidate_items):
                    values = [total[token] for token in tokens[item]]
                    metadata[position, relation_index*7:(relation_index+1)*7] = [
                        sum(values), max(values, default=0), sum(values)/len(seeds),
                        sum(recent[token] for token in tokens[item])/recent_count,
                        sum(decayed[token] for token in tokens[item])/decay_mass,
                        len(last1.intersection(tokens[item])), len(last3.intersection(tokens[item]))]
        additional = pd.DataFrame(np.column_stack([graph_features, temporal, metadata]), index=candidates.index,
                                  columns=[f'manual_co_{n}' for n in graph_names] +
                                          [f'manual_user_{n}' for n in temporal_names] + metadata_names)
        blocks = [result, additional]
        if self.config.get('extended_aggregates'):
            blocks.append(self.extended_features(candidates))
        return pd.concat(blocks, axis=1).astype('float32').copy()

    def extended_features(self, candidates):
        """Past-only lifetime, link concentration and token probability features.

        Missing graph links have age 999 (with an explicit support feature).
        Token masses split each event equally across its nonempty tokens.
        """
        names = ['item_observed_age', 'item_idle_days', 'item_recent7_fraction',
                 'item_recent3_fraction', 'item_daily_rate', 'co_nearest_age',
                 'co_squared_sum', 'co_support_fraction', 'co_max_decay7']
        kinds = ['entropy', 'concentration', 'probability', 'decay7_probability']
        names += [f'{relation}_{kind}' for relation in self.tokens for kind in kinds]
        values = np.zeros((len(candidates), len(names)), dtype='float32')
        if not hasattr(self, '_item_lifetime'):
            self._item_lifetime = self.history.groupby('item_id').date.agg(['min', 'max', 'count'])
        life = self._item_lifetime
        first = candidates.item_id.map((self.cutoff-life['min']).dt.days).fillna(999).to_numpy()
        idle = candidates.item_id.map((self.cutoff-life['max']).dt.days).fillna(999).to_numpy()
        count = candidates.item_id.map(life['count']).fillna(0).to_numpy()
        values[:, :5] = np.column_stack([first, idle,
            candidates.item_id.map(self.item_counts[7]).fillna(0).to_numpy()/np.maximum(count, 1),
            candidates.item_id.map(self.item_counts[3]).fillna(0).to_numpy()/np.maximum(count, 1),
            count/np.maximum(first, 1)])
        values[:, 5] = 999
        for user, positions in candidates.groupby('user_id', sort=False).indices.items():
            item_indices = self.item_index.get_indexer(candidates.iloc[positions].item_id)
            seeds, ages = self.histories.get(int(user), ([], []))
            if not len(seeds):
                continue
            links = self.graph[seeds][:, item_indices].toarray()
            mask = links > 0
            values[positions, 5] = np.where(mask, np.asarray(ages)[:, None], 999).min(axis=0)
            values[positions, 6] = np.square(links).sum(axis=0)
            values[positions, 7] = mask.sum(axis=0)/len(seeds)
            values[positions, 8] = (links*np.exp(-np.asarray(ages)[:, None]/7)).max(axis=0)
            for number, tokens in enumerate(self.tokens.values()):
                total, recent = Counter(), Counter()
                for seed, age in zip(seeds, ages):
                    for token in tokens[seed]:
                        total[token] += 1/len(tokens[seed])
                        recent[token] += np.exp(-age/7)/len(tokens[seed])
                mass, recent_mass = sum(total.values()), sum(recent.values())
                probabilities = np.array(list(total.values()))/max(mass, 1e-12)
                entropy = -np.sum(probabilities*np.log(np.maximum(probabilities, 1e-12)))
                concentration = np.square(probabilities).sum()
                for position, item in zip(positions, item_indices):
                    values[position, 9+number*4:13+number*4] = [entropy, concentration,
                        sum(total[t] for t in tokens[item])/max(mass, 1e-12),
                        sum(recent[t] for t in tokens[item])/max(recent_mass, 1e-12)]
        return pd.DataFrame(values, index=candidates.index, columns=['manual_v3_'+n for n in names])
