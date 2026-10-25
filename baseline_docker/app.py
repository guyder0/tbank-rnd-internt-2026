"""Train once on /data/train.csv, persist /data/model.cbm, then evaluate test.csv."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time
import traceback

from catboost import CatBoostRanker, Pool
import numpy as np
import pandas as pd

from baseline import aggregates, pipeline, refined
from baseline.pipeline import CATEGORY_COLUMNS, evaluate, history_before, log, targets
from baseline_docker import features
from baseline_docker.features import FeatureBuilder

DEFAULTS = Path(__file__).with_name('default_config.json')
INPUTS = ['train.csv','test.csv','users.csv','items.csv']
USER_COLUMNS = ['user_id','age','income','sex','kids_flg']
ITEM_COLUMNS = ['item_id','content_type','release_year','age_rating','for_kids',
                'genres','countries','directors','actors','studios']


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


def source_hashes():
    paths = [Path(__file__), Path(features.__file__), Path(pipeline.__file__),
             Path(aggregates.__file__), Path(refined.__file__)]
    return {p.name: sha256(p) for p in paths}


def load_config(path=None):
    defaults = json.loads(DEFAULTS.read_text())
    config = dict(defaults)
    if path:
        changes = json.loads(Path(path).read_text())
        if set(changes)-set(defaults):
            raise ValueError(f'Unknown config fields: {sorted(set(changes)-set(defaults))}')
        config.update(changes)
    if config['feature_columns'] != defaults['feature_columns'] or not config['extended_aggregates']:
        raise ValueError('Feature schema must match combined seed44 (191 features)')
    for key in ['sample_users','train_weeks','horizon_days','iterations','depth','thread_count',
                'feature_batch_users','neighbors','graph_history_limit','selection_period','bootstrap_iterations']:
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if not {7,30,90}.issubset(config['windows']):
        raise ValueError('Feature windows must include 7, 30, 90')
    if 10 not in config['metric_ks'] or any(type(k) is not int or k<1 for k in config['metric_ks']):
        raise ValueError('metric_ks must contain positive integers including 10')
    if not 0 < config['gpu_ram_part'] <= 1:
        raise ValueError('gpu_ram_part must be in (0,1]')
    return config


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


def choose_backend(requested, devices, config):
    if requested == 'cpu':
        return 'CPU', None
    # A failed CUDA initialization must not poison the long-running CPU process.
    probe = ("from catboost import CatBoostRanker; "
             f"m=CatBoostRanker(iterations=2,depth=2,task_type='GPU',devices={devices!r},"
             f"gpu_ram_part={config['gpu_ram_part']!r},verbose=False,allow_writing_files=False); "
             "m.fit([[0],[1],[0],[1]],[0,1,0,1],group_id=[0,0,1,1]); print('GPU preflight OK')")
    try:
        process = subprocess.run([sys.executable,'-c',probe],capture_output=True,text=True,timeout=45)
        if process.returncode == 0:
            log('GPU preflight OK')
            return 'GPU', None
        reason = (process.stderr or process.stdout).strip()
    except subprocess.TimeoutExpired:
        reason = 'GPU preflight exceeded 45 seconds'
    if requested == 'gpu':
        raise RuntimeError(f'GPU explicitly requested but unavailable: {reason}')
    log(f'GPU unavailable; training on CPU. Reason: {reason}')
    return 'CPU', reason


def batches(ids, size):
    for offset in range(0, len(ids), size):
        yield ids[offset:offset+size]


def snapshot(events, users, items, cutoff, selected, config, query_offset, training):
    history = history_before(events, cutoff)
    warm = np.intersect1d(selected, history.user_id.unique())
    truth = targets(events, warm, cutoff, config['horizon_days'])
    ids = np.array(sorted(truth), dtype='int64')
    if not len(ids):
        raise ValueError(f'No warm users with new positives at {cutoff.date()}')
    builder = FeatureBuilder(history, users, items, cutoff, ids, config)
    frames = []
    for part in batches(ids, config['feature_batch_users']):
        frame = builder.build(part)
        frame['label'] = [int(item in truth.get(int(user),set()))
                          for user,item in frame[['user_id','item_id']].itertuples(index=False,name=None)]
        if training:
            frame = frame.loc[frame.groupby('user_id').label.transform('sum').gt(0)].copy()
        frame['query_id'] = query_offset+frame.user_id
        frames.append(frame)
        log(f'{cutoff.date()}: features for batch of {len(part)} users')
    frame = pd.concat(frames,ignore_index=True).sort_values(['query_id','item_id']).reset_index(drop=True)
    if frame.empty:
        raise ValueError(f'No positive training candidates at {cutoff.date()}')
    manifest = {'cutoff':str(cutoff.date()),'rows':len(frame),'positive_pairs':int(frame.label.sum()),
                'query_users_with_candidates':int(frame.user_id.nunique()),'users_with_future_positives':len(truth)}
    log(manifest)
    return frame, truth, manifest


def end_to_end_ndcg(frame, scores, truth):
    order = np.lexsort((frame.item_id.to_numpy(), -np.asarray(scores), frame.user_id.to_numpy()))
    groups = frame.user_id.to_numpy()[order]
    starts = np.r_[0,np.flatnonzero(groups[1:] != groups[:-1])+1]
    ranks = np.arange(len(groups))-np.repeat(starts,np.diff(np.r_[starts,len(groups)]))
    discounted = frame.label.to_numpy()[order]/np.log2(ranks+2)
    discounted[ranks>=10] = 0
    dcg = np.add.reduceat(discounted, starts)
    ideal = np.array([sum(1/np.log2(np.arange(min(10,len(truth[int(user)])))+2)) for user in groups[starts]])
    return float(np.sum(dcg/ideal)/len(truth))


def fit_model(events, users, items, cutoff, config, backend, devices, run):
    rng = np.random.default_rng(config['sample_seed'])
    population = np.sort(events.user_id.unique())
    selected = np.sort(rng.choice(population,min(config['sample_users'],len(population)),replace=False))
    pd.Series(selected,name='user_id').to_csv(run/'training_users.csv',index=False)
    frames, manifests = [], []
    stride = int(events.user_id.max())+1
    for index in range(config['train_weeks']):
        split_cutoff = cutoff-pd.Timedelta(days=config['horizon_days']*(config['train_weeks']+1-index))
        frame, _, manifest = snapshot(events, users, items, split_cutoff, selected, config, index*stride, True)
        frames.append(frame); manifests.append(manifest)
    training = pd.concat(frames,ignore_index=True)
    validation, truth, manifest = snapshot(events, users, items,
        cutoff-pd.Timedelta(days=config['horizon_days']),selected,config,config['train_weeks']*stride,False)
    manifests.append(manifest)
    columns = config['feature_columns']
    def pool(frame):
        return Pool(frame[columns],frame.label,group_id=frame.query_id,cat_features=CATEGORY_COLUMNS)
    params = {'task_type':backend,'iterations':config['iterations'],'depth':config['depth'],
              'learning_rate':config['learning_rate'],'l2_leaf_reg':config['l2_leaf_reg'],
              'loss_function':'YetiRank','eval_metric':'NDCG:top=10','thread_count':config['thread_count'],
              'random_seed':config['seed'],'allow_writing_files':False}
    if backend == 'GPU':
        params.update(devices=devices,gpu_ram_part=config['gpu_ram_part'])
    model = CatBoostRanker(**params)
    log(f'Training {len(training)} rows, {len(columns)} features, backend={backend}, seed={config["seed"]}')
    model.fit(pool(training),eval_set=pool(validation),use_best_model=False,
              early_stopping_rounds=100,verbose=50,log_cout=sys.stdout,log_cerr=sys.stderr)
    write_json(run/'learning_curve.json',model.get_evals_result())
    write_json(run/'catboost_parameters.json',model.get_all_params())
    best_score, best_trees, selection = -1., 0, []
    for index, prediction in enumerate(model.staged_predict(validation[columns],eval_period=config['selection_period'])):
        trees = min((index+1)*config['selection_period'],model.tree_count_)
        value = end_to_end_ndcg(validation,prediction,truth)
        selection.append({'trees':trees,'ndcg@10':value})
        if value>best_score:
            best_score, best_trees = value, trees
    model.shrink(best_trees)
    log(f'Selected {best_trees} trees by validation end-to-end NDCG@10={best_score:.6f}')
    importance = model.get_feature_importance(type='PredictionValuesChange')
    pd.DataFrame({'feature':columns,'importance':importance}).sort_values('importance',ascending=False).to_csv(run/'feature_importance.csv',index=False)
    report = {'splits':manifests,'selection':selection,'validation_ndcg@10':best_score,
              'trees':best_trees,'sample_users':len(selected),'training_backend':backend}
    write_json(run/'training.json',report)
    return model, report


def evaluate_test(model, train, test, users, items, cutoff, config, run):
    # Test labels are read only here for scoring, after saving/committing the model.
    seen = train.groupby('user_id').item_id.agg(set).to_dict()
    truth = {}
    seen_pairs = 0
    for user,item in test[['user_id','item_id']].itertuples(index=False,name=None):
        if item in seen.get(user,set()):
            seen_pairs += 1
        else:
            truth.setdefault(int(user),set()).add(int(item))
    if not truth:
        raise ValueError('Test contains no new user-item pairs to evaluate')
    ids = np.array(sorted(truth),dtype='int64')
    builder = FeatureBuilder(train,users,items,cutoff,ids,config)
    history_counts = train.groupby('user_id').size()
    records = []
    coverage = set()
    recommendation_path = run/'recommendations.csv'
    written = False
    maximum_k = max(config['metric_ks'])
    for part in batches(ids,config['feature_batch_users']):
        frame = builder.build(part)
        batch_truth = {int(u):truth[int(u)] for u in part}
        # Ranker.predict in the pinned wheel uses CPU and has no task_type kwarg.
        scores = model.predict(frame[config['feature_columns']],thread_count=config['thread_count']) if len(frame) else np.array([])
        table = None
        for k in config['metric_ks']:
            _, individual = evaluate(frame,scores,batch_truth,k=k)
            individual = individual.rename(columns={'ndcg':f'ndcg@{k}','recall':f'recall@{k}','map':f'map@{k}'}).set_index('user_id')
            if table is None:
                table = individual
            else:
                table = table.join(individual.drop(columns='candidate_recall'),how='left')
        popularity = frame.item_count_30d.to_numpy() if len(frame) else np.array([])
        _, reference = evaluate(frame,popularity,batch_truth,k=10)
        table = table.join(reference.set_index('user_id')[['ndcg','recall','map']].add_prefix('popularity_'))
        table['history_items'] = history_counts.reindex(table.index,fill_value=0)
        table['is_warm'] = table.history_items.gt(0)
        table['future_positive_items'] = [len(truth[int(u)]) for u in table.index]
        table['candidates'] = frame.groupby('user_id').size().reindex(table.index,fill_value=0)
        table['precision@10'] = table['recall@10']*table.future_positive_items/10
        table['hit_rate@10'] = table['ndcg@10'].gt(0).astype(float)
        records.append(table)
        ranked = frame[['user_id','item_id']].assign(score=scores).sort_values(['user_id','score','item_id'],ascending=[True,False,True])
        ranked['rank'] = ranked.groupby('user_id').cumcount()+1
        ranked = ranked.loc[ranked['rank']<=maximum_k]
        coverage.update(ranked.loc[ranked['rank']<=10,'item_id'])
        ranked.to_csv(recommendation_path,index=False,mode='a' if written else 'w',header=not written)
        written = True
        log(f'Test: scored batch of {len(part)} users ({len(frame)} candidate pairs), inference=CPU')
    per_user = pd.concat(records).sort_index()
    assert len(per_user)==len(truth) and per_user.index.is_unique
    per_user.to_csv(run/'test_per_user.csv')
    metric_columns = [c for c in per_user if '@' in c or c.startswith('popularity_') or c=='candidate_recall']
    def metrics(part):
        if part.empty:
            return {'users':0,'metrics':None}
        return {'users':len(part),'metrics':{c:float(part[c].mean()) for c in metric_columns},
                'no_candidate_users':int(part.candidates.eq(0).sum()),
                'complete_retrieval_miss_users':int(part.candidate_recall.eq(0).sum()),
                'future_positive_items':int(part.future_positive_items.sum())}
    segments = pd.cut(per_user.history_items,bins=[-1,0,1,2,5,10,20,50,np.inf],
                      labels=['cold','1','2','3-5','6-10','11-20','21-50','>50'])
    by_history = {str(name):metrics(part) for name,part in per_user.groupby(segments,observed=True)}
    def bootstrap(values):
        rng = np.random.default_rng(config['seed'])
        return np.quantile([rng.choice(values,len(values),replace=True).mean()
                            for _ in range(config['bootstrap_iterations'])],[.025,.975]).tolist()
    report = {'target':'new observed user-item pair during test interval; previously seen pairs excluded',
              'primary_population':'all test users with new positives, including cold users',
              'overall':metrics(per_user),'warm':metrics(per_user.loc[per_user.is_warm]),
              'cold':metrics(per_user.loc[~per_user.is_warm]),'by_history':by_history,
              'seen_test_pairs_excluded':seen_pairs,'test_users_without_new_positives':int(test.user_id.nunique()-len(truth)),
              'catalogue_coverage@10':len(coverage)/len(items),'recommended_unique_items@10':len(coverage),
              'ndcg@10_ci95':bootstrap(per_user['ndcg@10'].to_numpy()),
              'delta_vs_popularity_ndcg@10':float((per_user['ndcg@10']-per_user.popularity_ndcg).mean()),
              'delta_vs_popularity_ci95':bootstrap((per_user['ndcg@10']-per_user.popularity_ndcg).to_numpy()),
              'ci_limitation':'User bootstrap conditional on fixed model and one interval; no correction for model selection',
              'inference_backend':'CPU','popularity_reference_k':10,
              'cold_policy':'original popularity/genre candidates without personal history'}
    return report


def validate_model(model, metadata, cutoff, config, fingerprints):
    if model.feature_names_ != config['feature_columns']:
        raise ValueError('Saved model feature schema differs from combined (191 columns)')
    if metadata:
        if metadata.get('schema_version') != 1:
            raise ValueError('Unsupported model.json schema')
        if pd.Timestamp(metadata['trained_before'])>cutoff:
            raise ValueError('Saved model used data later than this test cutoff; refusing temporal leakage')
        if metadata.get('source_sha256') and metadata['source_sha256'] != source_hashes():
            raise ValueError('Feature/runtime code changed since model training; use the matching image or a separate data directory')
        training_hashes = metadata.get('input_sha256',{})
        differences = [k for k in ['train.csv','users.csv','items.csv'] if k in training_hashes and training_hashes[k] != fingerprints[k]]
        if differences:
            log(f'Existing model reused with changed input history/metadata: {differences}; no retraining')
    else:
        log('Found external model.cbm without model.json: training provenance/cutoff cannot be verified')


class Tee:
    def __init__(self, stream, logfile):
        self.stream, self.logfile = stream, logfile
    def write(self, value):
        self.stream.write(value); self.logfile.write(value)
        self.flush()
    def flush(self):
        self.stream.flush(); self.logfile.flush()
    def isatty(self):
        return False


@contextmanager
def directory_lock(directory):
    with (directory/'.baseline.lock').open('a') as stream:
        try:
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another container/process is using this data directory') from None
        yield


def execute(args, directory, run):
    started = time.monotonic()
    model_path = directory/'model.cbm'
    metadata_path = directory/'model.json'
    metadata = json.loads(metadata_path.read_text()) if model_path.exists() and metadata_path.exists() else None
    # Reuse saved feature and evaluation definitions even when only weights are loaded.
    config = metadata['config'] if metadata and args.config is None else load_config(args.config)
    if metadata and args.config is not None and config != metadata['config']:
        raise ValueError('Config differs from saved model; omit --config to reuse its config')
    write_json(run/'config.json',config)
    train, test, users, items, cutoff, end, fingerprints = read_inputs(directory,config)
    log(f'Input train={len(train)} pairs, test={len(test)} pairs; test=[{cutoff.date()},{end.date()})')
    write_json(run/'inputs.json',{'sha256':fingerprints,'train_rows':len(train),'test_rows':len(test),
                                 'test_cutoff':str(cutoff.date()),'test_end_exclusive':str(end.date())})
    backend_reason = None
    if model_path.exists():
        log(f'Found {model_path}; skipping training and GPU initialization')
        if metadata and sha256(model_path) != metadata['model_sha256']:
            raise ValueError('model.cbm checksum does not match model.json')
        model = CatBoostRanker().load_model(str(model_path))
        validate_model(model,metadata,cutoff,config,fingerprints)
        action = 'loaded'
    else:
        earliest = cutoff-pd.Timedelta(days=config['horizon_days']*(config['train_weeks']+1))
        if train.date.min() >= earliest:
            raise ValueError(f'Training requires history before {earliest.date()}; '
                             'supply a longer train.csv or explicitly reduce train_weeks')
        backend, backend_reason = choose_backend(args.device,args.devices,config)
        model, training = fit_model(train,users,items,cutoff,config,backend,args.devices,run)
        metadata = {'schema_version':1,'origin':'trained_by_baseline_docker','config':config,
                    'trained_before':str(cutoff.date()),'source_sha256':source_hashes(),
                    'input_sha256':{k:v for k,v in fingerprints.items() if k!='test.csv'},
                    'training':training,'training_backend':backend,'gpu_fallback_reason':backend_reason}
        # Metadata first; atomic model rename is the final commit marker.
        with tempfile.TemporaryDirectory(prefix='.model-pending-',dir=directory) as temporary:
            pending = Path(temporary)/'model.cbm'
            model.save_model(str(pending))
            metadata['model_sha256'] = sha256(pending)
            write_json(metadata_path,metadata)
            pending.replace(model_path)
        action = 'trained'
        log(f'Saved model to {model_path} before evaluating test labels')
    evaluation_start = time.monotonic()
    report = evaluate_test(model,train,test,users,items,cutoff,config,run)
    report.update(model_action=action,model_sha256=sha256(model_path),trees=model.tree_count_,
                  training_backend=metadata.get('training_backend') if metadata else model.get_all_params().get('task_type'),
                  model_provenance_verified=metadata is not None,test_cutoff=str(cutoff.date()),test_end_exclusive=str(end.date()),
                  evaluation_seconds=time.monotonic()-evaluation_start,total_seconds=time.monotonic()-started,
                  peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                  versions={name:version(name) for name in ['catboost','numpy','pandas','scipy']})
    write_json(run/'metrics.json',report)
    write_json(directory/'metrics.json',report)
    log('Detailed test metrics:\n'+json.dumps(report,ensure_ascii=False,indent=2))
    log(f'Artifacts: {run}; latest metrics: {directory/"metrics.json"}')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=Path('/data'))
    parser.add_argument('--device',choices=['auto','cpu','gpu'],default='auto')
    parser.add_argument('--devices',default='0')
    parser.add_argument('--config',type=Path)
    args = parser.parse_args()
    directory = args.data_dir.resolve()
    if not directory.is_dir():
        parser.error(f'Data directory does not exist: {directory}; mount it at /data')
    with directory_lock(directory):
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        run = directory/'runs'/stamp
        run.mkdir(parents=True)
        status = {'status':'running','started_utc':stamp,'source_sha256':source_hashes()}
        write_json(run/'status.json',status)
        with (run/'run.log').open('w',buffering=1) as logfile:
            stdout, stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = Tee(stdout,logfile), Tee(stderr,logfile)
            try:
                report = execute(args,directory,run)
                status.update(status='completed',model_action=report['model_action'])
            except BaseException as error:
                status.update(status='failed',error=repr(error))
                traceback.print_exc()
                raise
            finally:
                write_json(run/'status.json',status)
                sys.stdout, sys.stderr = stdout, stderr


if __name__ == '__main__':
    main()
