"""Prepare the mounted raw CSV directory from the original KION archive."""
import argparse
import json
from pathlib import Path
import tempfile
import zipfile

import pandas as pd


def prepare(archive, output, cutoff='2021-08-16', horizon_days=7):
    output = Path(output).resolve()
    cutoff = pd.Timestamp(cutoff)
    if cutoff.tz is not None or cutoff != cutoff.normalize() or horizon_days<1:
        raise ValueError('Use a calendar cutoff and positive horizon_days')
    if output.exists():
        raise ValueError('Output directory already exists; choose a new directory')
    output.parent.mkdir(parents=True,exist_ok=True)
    end = cutoff+pd.Timedelta(days=horizon_days)
    counts = {'train':0,'test':0}
    with tempfile.TemporaryDirectory(prefix='.kion-prepare-',dir=output.parent) as temporary:
        staged = Path(temporary)/'data';staged.mkdir()
        with zipfile.ZipFile(archive) as source:
            with source.open('data_original/interactions.csv') as stream:
                for chunk in pd.read_csv(stream,usecols=['user_id','item_id','last_watch_dt'],chunksize=250000):
                    dates = pd.to_datetime(chunk.last_watch_dt,errors='raise')
                    for name,mask in [('train',dates<cutoff),('test',(dates>=cutoff)&(dates<end))]:
                        selected = chunk.loc[mask]
                        if len(selected):
                            selected.to_csv(staged/f'{name}.csv',mode='a' if counts[name] else 'w',
                                            index=False,header=not counts[name])
                            counts[name] += len(selected)
            for name in ['users.csv','items.csv']:
                with source.open('data_original/'+name) as stream:
                    (staged/name).write_bytes(stream.read())
        if not all(counts.values()):
            raise ValueError('The requested split has empty train or test')
        (staged/'dataset.json').write_text(json.dumps({'test_cutoff':str(cutoff.date()),
            'test_end_exclusive':str(end.date())},indent=2)+'\n')
        staged.rename(output)
    print(f'Prepared {output}: {counts}')
    return counts


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--test-cutoff',default='2021-08-16')
    parser.add_argument('--horizon-days',type=int,default=7)
    args=parser.parse_args()
    prepare(args.archive,args.output,args.test_cutoff,args.horizon_days)
