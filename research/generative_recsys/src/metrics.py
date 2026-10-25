"""Ranking metrics and paired uncertainty; no neural dependencies."""
import numpy as np
import pandas as pd
from baseline.pipeline import evaluate


def evaluate_lists(queries, lists, k):
    rows = [(q['user_id'], int(item), -rank) for q, result in zip(queries, lists)
            for rank, item in enumerate(result)]
    frame = pd.DataFrame(rows, columns=['user_id','item_id','score'])
    truth = {q['user_id']: set(q['positives']) for q in queries}
    return evaluate(frame, frame.score.to_numpy(), truth, k)


def paired_delta(first, reference, seed):
    left = first.set_index('user_id').sort_index()
    right = reference.set_index('user_id').sort_index()
    if not left.index.equals(right.index):
        raise ValueError('Different metric populations')
    values = (left.ndcg-right.ndcg).to_numpy()
    rng = np.random.default_rng(seed)
    bootstrap = [float(rng.choice(values, len(values), replace=True).mean()) for _ in range(1000)]
    return {'ndcg_delta': float(values.mean()), 'ci95': np.quantile(bootstrap,[.025,.975]).tolist()}

