"""Train TF-IDF/RQ-VAE/SID Transformer once, then evaluate mounted external test."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import resource
import sys
import tempfile
import time
import traceback

import numpy as np
import pandas as pd
import torch

from baseline.pipeline import log
from generative_docker.io import directory_lock, read_inputs, sha256, write_json
from research.generative_recsys.src.data import prepare_queries
from research.generative_recsys.src.decoding import SIDTrie, beam_search
from research.generative_recsys.src.metrics import evaluate_lists
from research.generative_recsys.src.model import SIDGenerator, decoder_callback
from research.generative_recsys.src.tokenizer import content_embeddings, fit_codes, item_texts
from research.generative_recsys.src.training import train as train_generator


class Tee:
    def __init__(self,*streams):self.streams=streams
    def write(self,value):
        for stream in self.streams:stream.write(value);stream.flush()
    def flush(self):
        for stream in self.streams:stream.flush()
    def isatty(self):return False


def load_config(path=None):
    config=json.loads(Path(__file__).with_name('default_config.json').read_text())
    if path:
        changes=json.loads(Path(path).read_text())
        if set(changes)-set(config):raise ValueError('Unknown configuration keys')
        config.update(changes)
    if config['embedding_backend']!='tfidf' or config['tokenizer']!='rqvae' or config['random_codes']:
        raise ValueError('This container implements the measured TF-IDF + RQ-VAE recipe')
    for key in ['sample_users','train_weeks','horizon_days','epochs','patience','beam_width','batch_size',
                'threads','max_history','code_width','code_levels','embedding_dim','rqvae_epochs',
                'rqvae_batch_size','rqvae_latent_dim','kmeans_iterations','bootstrap_iterations','d_model','heads','layers']:
        if type(config[key]) is not int or config[key]<1:raise ValueError(f'{key} must be a positive integer')
    if config['horizon_days']!=7:raise ValueError('Query preparation requires a 7-day horizon')
    if config['d_model']%config['heads']:raise ValueError('d_model must be divisible by heads')
    for key in ['learning_rate','rqvae_learning_rate']:
        if not np.isfinite(config[key]) or config[key]<=0:raise ValueError(f'Invalid {key}')
    if config['top_k']!=10 or 10 not in config['metric_ks'] or any(type(k)is not int or k<1 for k in config['metric_ks']):
        raise ValueError('Require top_k=10 and positive metric_ks including 10')
    if config['beam_width']<max(config['metric_ks']):raise ValueError('beam_width must cover metric_ks')
    if config['rqvae_max_steps'] is not None:raise ValueError('Full recipe cannot limit tokenizer steps')
    return config


def source_hashes():
    root=Path(__file__).resolve().parents[1]
    paths=[Path(__file__),Path(__file__).with_name('io.py'),root/'baseline/pipeline.py']
    paths+=list((root/'research/generative_recsys/src').glob('*.py'))
    return {str(path.relative_to(root)):sha256(path) for path in paths}


def choose_device(requested):
    if requested=='cpu':return torch.device('cpu')
    try:
        if not torch.cuda.is_available():raise RuntimeError('CUDA is not available')
        _=torch.ones((2,2),device='cuda')@torch.ones((2,2),device='cuda')
        torch.cuda.synchronize()
        return torch.device('cuda')
    except (RuntimeError,AssertionError) as error:
        if requested in ['gpu','cuda']:raise RuntimeError(f'Explicit GPU request failed: {error}') from error
        log(f'GPU unavailable; using CPU: {error}')
        return torch.device('cpu')


def fit_bundle(events,items,cutoff,config,run,device):
    horizon=pd.Timedelta(days=7)
    cutoffs=[cutoff-horizon*(config['train_weeks']+1-i) for i in range(config['train_weeks']+1)]
    population=np.sort(events.user_id.unique())
    selected=np.sort(np.random.default_rng(config['sample_seed']).choice(population,min(config['sample_users'],len(population)),replace=False))
    pd.Series(selected,name='user_id').to_csv(run/'training_users.csv',index=False)
    catalog,splits=prepare_queries(events,items,selected,cutoffs,config['top_k'])
    if len(catalog)<3:raise ValueError('Need at least three historical catalog items')
    embeddings,_=content_embeddings(item_texts(catalog),config,'cpu')
    codebook=fit_codes(catalog.item_id.to_numpy(),embeddings,config,output=run/'tokenizer',device=device,log=log)
    prepared={'codebook':codebook,'splits':splits,'config':config,'source':{'synthetic':False}}
    training=train_generator(prepared,config,run,device,log)
    checkpoint=torch.load(run/'model.pt',map_location='cpu',weights_only=True)
    bundle={'schema_version':1,'checkpoint':checkpoint,
            'item_ids':torch.as_tensor(codebook['item_ids']),'codes':torch.as_tensor(codebook['codes'])}
    write_json(run/'training_splits.json',[{k:v for k,v in split.items() if k!='queries'}|{'queries':len(split['queries'])} for split in splits])
    return bundle,training


def save_bundle(bundle,directory,metadata):
    if (directory/'sid_model.pt').exists() or (directory/'sid_model.json').exists():
        raise ValueError('Existing SID model files would be overwritten')
    with tempfile.TemporaryDirectory(prefix='.sid-pending-',dir=directory) as temporary:
        pending=Path(temporary)/'sid_model.pt';torch.save(bundle,pending)
        metadata={**metadata,'model_sha256':sha256(pending)}
        write_json(directory/'sid_model.json',metadata)
        pending.replace(directory/'sid_model.pt')
    return metadata


def load_model(bundle,device):
    checkpoint=bundle['checkpoint'];config=checkpoint['config']
    ids=bundle['item_ids'].cpu().numpy();codes=bundle['codes'].cpu().numpy()
    if codes.ndim!=2 or len(ids)!=len(codes) or len(np.unique(ids))!=len(ids):raise ValueError('Invalid SID catalog')
    if len(np.unique(codes,axis=0))!=len(ids) or np.any(codes<3) or np.any(codes>=checkpoint['vocab_size']):
        raise ValueError('Invalid or duplicate SID codes')
    if codes.shape[1]!=checkpoint['sid_length']:raise ValueError('Invalid SID length')
    model=SIDGenerator(checkpoint['vocab_size'],checkpoint['sid_length'],config).to(device).eval()
    model.load_state_dict(checkpoint['state_dict'])
    return model,{int(item):code for item,code in zip(ids,codes)},SIDTrie(ids,codes)


def evaluate_test(bundle,events,test,cutoff,config,run,device):
    model,mapping,trie=load_model(bundle,device)
    torch.set_num_threads(config['threads'])
    # Keep personal histories only for requested users; global popularity still
    # uses all train users. Avoid one DataFrame per user across the entire KION.
    local=events[events.user_id.isin(test.user_id.unique())].sort_values(['user_id','date','item_id']).copy()
    local['age']=(cutoff-local.date).dt.days
    histories={int(user):(part.item_id.to_numpy(dtype='int64',copy=True),part.age.to_numpy(dtype='int64',copy=True))
               for user,part in local.groupby('user_id',sort=False)}
    seen={user:set(map(int,values[0])) for user,values in histories.items()}
    truth={};excluded=0
    for user,item in test[['user_id','item_id']].itertuples(index=False,name=None):
        if item in seen.get(user,set()):excluded+=1
        else:truth.setdefault(int(user),set()).add(int(item))
    if not truth:raise ValueError('No new positive user-item pairs in test')
    recent=events[events.date>=cutoff-pd.Timedelta(days=30)]
    popularity=recent.groupby('item_id').size().rename('n').reset_index().sort_values(['n','item_id'],ascending=[False,True]).item_id.tolist()
    queries=[];recommendations=[];references=[];rows=[];warm=[];counts=[];ceilings=[]
    maximum=max(config['metric_ks'])
    known=set(mapping)
    with torch.inference_mode():
        for number,(user,positives) in enumerate(sorted(truth.items())):
            history=histories.get(user)
            query={'user_id':user,'positives':sorted(positives),
                'items':history[0] if history is not None else np.array([],dtype='int64'),
                'ages':history[1] if history is not None else np.array([],dtype='int64')}
            popular=[int(item) for item in popularity if item not in seen.get(user,set())][:maximum]
            is_warm=history is not None
            if is_warm:
                callback=decoder_callback(model,query,mapping,config,device)
                generated=beam_search(callback,trie,config['beam_width'],maximum,query['items'])
                result=[int(item) for item,score in generated]
                rows.extend({'user_id':user,'item_id':int(item),'rank':rank,'score':float(score)} for rank,(item,score) in enumerate(generated,1))
            else:
                result=popular
                rows.extend({'user_id':user,'item_id':item,'rank':rank,'score':float(-rank)} for rank,item in enumerate(result,1))
            queries.append(query);recommendations.append(result);references.append(popular)
            warm.append(is_warm);counts.append(len(query['items']))
            ceilings.append(len(positives&known)/len(positives))
            if (number+1)%500==0:log(f'Test generation {number+1}/{len(truth)}')
    parts=[]
    for k in config['metric_ks']:
        _,individual=evaluate_lists(queries,[r[:k] for r in recommendations],k)
        individual=individual.set_index('user_id')[['ndcg','recall','map']].rename(columns={key:f'{key}@{k}' for key in ['ndcg','recall','map']})
        parts.append(individual)
    per_user=pd.concat(parts,axis=1)
    _,reference=evaluate_lists(queries,[r[:10] for r in references],10)
    per_user=per_user.join(reference.set_index('user_id')[['ndcg','recall','map']].add_prefix('popularity_'))
    per_user['is_warm']=warm;per_user['history_items']=counts;per_user['catalog_recall']=ceilings
    per_user['precision@10']=[len(set(r[:10])&truth[q['user_id']])/10 for q,r in zip(queries,recommendations)]
    per_user['hit_rate@10']=per_user['ndcg@10'].gt(0).astype(float)
    per_user.to_csv(run/'test_per_user.csv')
    pd.DataFrame(rows,columns=['user_id','item_id','rank','score']).to_csv(run/'recommendations.csv',index=False)
    columns=[c for c in per_user if '@' in c or c.startswith('popularity_') or c=='catalog_recall']
    def summary(part):
        return {'users':len(part),'metrics':{c:float(part[c].mean()) for c in columns} if len(part) else None}
    rng=np.random.default_rng(config['seed'])
    differences=(per_user['ndcg@10']-per_user.popularity_ndcg).to_numpy()
    interval=np.quantile([rng.choice(differences,len(differences),replace=True).mean() for _ in range(config['bootstrap_iterations'])],[.025,.975]).tolist()
    values=per_user['ndcg@10'].to_numpy()
    ndcg_interval=np.quantile([rng.choice(values,len(values),replace=True).mean() for _ in range(config['bootstrap_iterations'])],[.025,.975]).tolist()
    bins=pd.cut(per_user.history_items,[-1,0,1,2,5,10,20,50,np.inf],labels=['cold','1','2','3-5','6-10','11-20','21-50','>50'])
    return {'overall':summary(per_user),'warm':summary(per_user[per_user.is_warm]),'cold':summary(per_user[~per_user.is_warm]),
        'by_history':{str(name):summary(part) for name,part in per_user.groupby(bins,observed=True)},
        'ndcg@10_ci95':ndcg_interval,
        'primary_population':'all test users with new positives; cold users receive train popularity',
        'delta_vs_popularity_ndcg@10':float(differences.mean()),'delta_vs_popularity_ci95':interval,
        'seen_test_pairs_excluded':excluded,'test_users_without_new_positives':int(test.user_id.nunique()-len(truth)),
        'short_list_users':sum(len(r)<10 for r in recommendations),'inference_backend':str(device),
        'catalog_items':len(mapping),'catalog_policy':'warm: fixed before earliest training cutoff, unseen positives retained; cold: full train popularity',
        'limitation':'one interval, fixed model; user bootstrap does not account for model selection'}


def execute(args,directory,run):
    path=directory/'sid_model.pt';metadata_path=directory/'sid_model.json'
    if path.exists()!=metadata_path.exists():raise ValueError('Incomplete SID model pair; restore both files or use another data directory')
    metadata=json.loads(metadata_path.read_text()) if path.exists() else None
    config=metadata['config'] if metadata and args.config is None else load_config(args.config)
    if metadata and config!=metadata['config']:raise ValueError('Config differs from saved model')
    write_json(run/'config.json',config)
    events,test,users,items,cutoff,end,fingerprints=read_inputs(directory,config)
    write_json(run/'inputs.json',{'sha256':fingerprints,'test_cutoff':str(cutoff.date()),'test_end_exclusive':str(end.date())})
    if metadata:
        if sha256(path)!=metadata['model_sha256']:raise ValueError('SID model checksum mismatch')
        if cutoff<pd.Timestamp(metadata['trained_before']):raise ValueError('Test overlaps training/validation selection period')
        bundle=torch.load(path,map_location='cpu',weights_only=True)
        if bundle['checkpoint']['config']!=config:raise ValueError('Bundle and metadata config mismatch')
        action='loaded';log('Loading saved SID model; no training or training-GPU initialization')
    else:
        device=choose_device(args.device)
        bundle,training=fit_bundle(events,items,cutoff,config,run,device)
        metadata=save_bundle(bundle,directory,{'schema_version':1,'origin':'trained_by_generative_docker',
            'config':config,'trained_before':str(cutoff.date()),'training_backend':str(device),
            'input_sha256':{k:v for k,v in fingerprints.items() if k!='test.csv'},'training':training,'source_sha256':source_hashes()})
        action='trained';log('Saved model bundle before test evaluation')
    inference=choose_device(args.inference_device)
    report=evaluate_test(bundle,events,test,cutoff,config,run,inference)
    report.update(model_action=action,model_sha256=sha256(path),training_backend=metadata['training_backend'],
        test_cutoff=str(cutoff.date()),test_end_exclusive=str(end.date()),
        versions={name:version(name) for name in ['torch','numpy','pandas','scipy']},
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    write_json(run/'metrics.json',report);write_json(directory/'sid_metrics.json',report)
    log(json.dumps(report,ensure_ascii=False,indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=Path('/data'))
    parser.add_argument('--device',choices=['auto','cpu','gpu','cuda'],default='auto')
    parser.add_argument('--inference-device',choices=['auto','cpu','gpu','cuda'],default='cpu')
    parser.add_argument('--config',type=Path)
    args=parser.parse_args();directory=args.data_dir.resolve()
    if not directory.is_dir():parser.error('Data directory must exist and be mounted at /data')
    with directory_lock(directory):
        run=directory/'runs'/('sid_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
        run.mkdir(parents=True)
        status={'status':'running','source_sha256':source_hashes()}
        started=time.monotonic()
        write_json(run/'status.json',status)
        old_stdout,old_stderr=sys.stdout,sys.stderr
        with (run/'run.log').open('w',buffering=1) as logfile:
            sys.stdout,sys.stderr=Tee(old_stdout,logfile),Tee(old_stderr,logfile)
            try:
                report=execute(args,directory,run)
                status.update(status='completed',model_action=report['model_action'])
            except BaseException as error:
                status.update(status='failed',error=repr(error));traceback.print_exc();raise
            finally:
                status['elapsed_seconds']=time.monotonic()-started
                write_json(run/'status.json',status)
                sys.stdout,sys.stderr=old_stdout,old_stderr


if __name__=='__main__':main()
