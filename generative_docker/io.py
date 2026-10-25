"""Raw CSV contract and atomic writes, matching baseline_docker (no ranker dependency)."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
INPUTS = ['train.csv','test.csv','users.csv','items.csv']
USER_COLUMNS = ['user_id','age','income','sex','kids_flg']
ITEM_COLUMNS = ['item_id','content_type','release_year','age_rating','for_kids','genres','countries','directors','actors','studios','title']


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()

def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temporary.replace(path)

def check_ids(frame, columns, name):
    for column in columns:
        values = pd.to_numeric(frame[column], errors='raise')
        if values.isna().any() or not np.isfinite(values).all() or (values<0).any() or (values%1!=0).any():
            raise ValueError(f'{name}: {column} must contain nonnegative integer IDs')
        frame[column] = values.astype('int64')

def read_interactions(path):
    frame = pd.read_csv(path)
    date_column = 'last_watch_dt' if 'last_watch_dt' in frame else 'date'
    required = ['user_id','item_id',date_column]
    if set(required)-set(frame):
        raise ValueError(f'{path.name}: required columns user_id, item_id, last_watch_dt (or date)')
    frame = frame[required].copy().rename(columns={date_column:'date'})
    if frame.empty:
        raise ValueError(f'{path.name} is empty')
    check_ids(frame, ['user_id','item_id'], path.name)
    frame['date'] = pd.to_datetime(frame.date, errors='raise')
    if frame.date.isna().any() or frame.date.dt.tz is not None or not frame.date.equals(frame.date.dt.normalize()):
        raise ValueError(f'{path.name}: dates must be timezone-free calendar dates (YYYY-MM-DD)')
    if frame.duplicated(['user_id','item_id']).any():
        raise ValueError(f'{path.name}: expected one aggregated row per user-item; duplicate pairs found')
    return frame

def read_inputs(directory, config):
    missing = [name for name in INPUTS if not (directory/name).is_file()]
    if missing:
        raise ValueError(f'Missing required input files in {directory}: {", ".join(missing)}')
    train = read_interactions(directory/'train.csv')
    test = read_interactions(directory/'test.csv')
    users = pd.read_csv(directory/'users.csv')
    items = pd.read_csv(directory/'items.csv')
    for frame, columns, name, key in [(users,USER_COLUMNS,'users.csv','user_id'),
                                       (items,ITEM_COLUMNS,'items.csv','item_id')]:
        if set(columns)-set(frame):
            raise ValueError(f'{name}: missing columns {sorted(set(columns)-set(frame))}')
        check_ids(frame, [key], name)
        if frame[key].duplicated().any():
            raise ValueError(f'{name}: duplicate {key}')
    users, items = users[USER_COLUMNS].copy(), items[ITEM_COLUMNS].copy()
    for name, frame in [('train.csv',train),('test.csv',test)]:
        if not frame.item_id.isin(items.item_id).all():
            raise ValueError(f'{name}: item IDs absent from items.csv')
    dataset_path = directory/'dataset.json'
    dataset = json.loads(dataset_path.read_text()) if dataset_path.exists() else {}
    if set(dataset)-{'test_cutoff','test_end_exclusive'}:
        raise ValueError('dataset.json supports only test_cutoff and test_end_exclusive')
    cutoff = pd.Timestamp(dataset.get('test_cutoff', test.date.min()))
    end = pd.Timestamp(dataset.get('test_end_exclusive', cutoff+pd.Timedelta(days=config['horizon_days'])))
    if cutoff.tz is not None or cutoff != cutoff.normalize() or end.tz is not None or end != end.normalize():
        raise ValueError('Dataset boundaries must be calendar dates without timezone')
    if end != cutoff+pd.Timedelta(days=config['horizon_days']):
        raise ValueError('Test interval must match horizon_days')
    if train.date.max() >= cutoff or test.date.min() < cutoff or test.date.max() >= end:
        raise ValueError('Require train dates < test_cutoff <= test dates < test_end_exclusive')
    fingerprints = {name:sha256(directory/name) for name in INPUTS}
    return train, test, users, items, cutoff, end, fingerprints

@contextmanager
def directory_lock(directory):
    with (directory/'.sid.lock').open('a') as stream:
        try:
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another container/process is using this data directory') from None
        yield
