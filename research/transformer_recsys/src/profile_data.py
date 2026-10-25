"""Profile strict-past KION histories without inspecting future labels.

Run from repository root:
    baseline/.venv/bin/python -m research.transformer_recsys.src.profile_data
"""
import json

import pandas as pd

from baseline.pipeline import ROOT, history_before, read_data


def main():
    events, _, items = read_data(ROOT / 'KION_DATASET')
    records = []
    for date in ['2021-07-19', '2021-07-26', '2021-08-02', '2021-08-09']:
        history = history_before(events, pd.Timestamp(date))
        lengths = history.groupby('user_id').size()
        daily = history.groupby(['user_id', 'date']).size()
        records.append({
            'cutoff': date,
            'history_rows': len(history),
            'users_with_history': len(lengths),
            'history_length_quantiles': {
                str(q): float(lengths.quantile(q)) for q in [.5, .9, .95, .99]
            },
            'users_more_than_128': int((lengths > 128).sum()),
            'events_in_tied_user_days_fraction': float(daily[daily > 1].sum() / len(history)),
        })
    report = {
        'source': 'KION_DATASET/data_original.zip',
        'meaning': 'One row per user-item, last observed date, not session events',
        'interactions': len(events),
        'items_in_metadata': len(items),
        'cutoffs': records,
    }
    destination = ROOT / 'research/transformer_recsys/kion_profile.json'
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
