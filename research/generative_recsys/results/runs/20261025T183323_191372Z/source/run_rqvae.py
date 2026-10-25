"""Train ONLY RQ-VAE on existing embeddings, with full-catalog diagnostics."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np

from research.generative_recsys.run import HOME, ROOT, Tee, digest, log, preflight, write_json
from research.generative_recsys.src.rqvae import train_rqvae


def plot_diagnostics(output):
    os.environ.setdefault('MPLCONFIGDIR',str(output/'plot_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = json.loads((output/'tokenizer/diagnostics.json').read_text())
    epochs = [row['epoch'] for row in rows]
    figure,axes = plt.subplots(2,2,figsize=(11,7),layout='constrained')
    axes[0,0].plot(epochs,[r['catalog_losses']['reconstruction'] for r in rows],label='RQ-VAE')
    axes[0,0].axhline(rows[0]['mean_baseline_mse'],color='gray',linestyle='--',label='Mean-vector baseline')
    axes[0,0].set(title='Reconstruction squared L2',yscale='log');axes[0,0].legend()
    for level in range(len(rows[0]['usage'])):
        axes[0,1].plot(epochs,[r['usage'][level]['used'] for r in rows],label=f'Level {level+1}')
    axes[0,1].set(title='Occupied codes');axes[0,1].legend()
    for key in ['input_variance','latent_variance','reconstruction_variance']:
        axes[1,0].plot(epochs,[max(r[key],1e-15) for r in rows],label=key)
    axes[1,0].set(title='Variance (different representation spaces)',yscale='log');axes[1,0].legend()
    for key in ['codebook','commitment']:
        axes[1,1].plot(epochs,[max(r['catalog_losses'][key],1e-15) for r in rows],label=key)
    axes[1,1].set(title='Quantization losses',yscale='log');axes[1,1].legend()
    for axis in axes.flat:
        axis.set_xlabel('Epoch');axis.grid(alpha=.2)
    figure.savefig(output/'dynamics.png',dpi=150);plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',type=Path,default=HOME/'artifacts/20261025T180753_899640Z')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--epochs',type=int,default=1000)
    parser.add_argument('--learning-rate',type=float,default=.1)
    parser.add_argument('--optimization',choices=['joint','alternating'],default='joint')
    parser.add_argument('--diagnostics-every',type=int,default=10)
    parser.add_argument('--max-steps',type=int,help='Limit optimizer steps for a CPU launch check')
    parser.add_argument('--smoke',action='store_true',help='128 saved embeddings, four CPU optimizer steps')
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    if Path(sys.prefix).resolve() != (ROOT/'baseline/.venv').resolve():
        parser.error('Use baseline/.venv/bin/python')
    if min(args.epochs,args.diagnostics_every) < 1 or not np.isfinite(args.learning_rate) or args.learning_rate<=0:
        parser.error('Epochs, learning rate and diagnostics interval must be positive')
    if args.max_steps is not None and args.max_steps<1:
        parser.error('--max-steps must be positive')
    if args.smoke and args.device!='cpu':
        parser.error('--smoke requires --device cpu')
    source = args.prepared.resolve()
    config = json.loads((source/'config.json').read_text())
    config.update(tokenizer='rqvae',rqvae_epochs=args.epochs,rqvae_learning_rate=args.learning_rate,
        rqvae_diagnostics_every=args.diagnostics_every,rqvae_max_steps=args.max_steps,
        rqvae_optimization=args.optimization)
    if args.smoke:
        config.update(rqvae_epochs=3,rqvae_max_steps=4,rqvae_batch_size=64,rqvae_diagnostics_every=1)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = (args.output or HOME/'artifacts/rqvae_only'/stamp).resolve()
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'config.json',config)
    copies = output/'source';copies.mkdir()
    for path in [Path(__file__),HOME/'src/rqvae.py',HOME/'run.py']:
        shutil.copy2(path,copies/path.name)
    status = {'status':'running','stage':'preflight','device':args.device,'smoke':args.smoke,
        'command':sys.argv,'python':sys.version,'prepared_source':str(source),
        'source_sha256':{p.name:digest(p) for p in copies.iterdir()},
        'dependencies':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()}}
    started = time.monotonic()
    old_stdout,old_stderr = sys.stdout,sys.stderr
    with (output/'run.log').open('w',buffering=1) as logfile:
        sys.stdout,sys.stderr = Tee(old_stdout,logfile),Tee(old_stderr,logfile)
        try:
            log(f'RQ-VAE-only run: {output}')
            write_json(output/'status.json',status)
            device = preflight(args.device)
            path = source/'item_embeddings.npy'
            fingerprint = digest(path)
            manifest = json.loads((source/'prepare_manifest.json').read_text())
            if manifest.get('embeddings_sha256') != fingerprint:
                raise ValueError('Prepared embeddings checksum mismatch')
            embeddings = np.load(path,allow_pickle=False)
            if args.smoke:
                selected = np.random.default_rng(config['seed']).choice(len(embeddings),min(128,len(embeddings)),replace=False)
                embeddings = embeddings[selected]
                np.save(output/'selected_rows.npy',selected)
            write_json(output/'input_manifest.json',{'source':str(path),'sha256':fingerprint,
                'embedding':manifest['embedding'],'shape':list(embeddings.shape),
                'sampled':args.smoke,'diagnostics':'same catalog used for training; not held-out generalization'})
            status['stage']='train';write_json(output/'status.json',status)
            codes,_,report = train_rqvae(embeddings,config,output/'tokenizer',device,log)
            np.save(output/'codes.npy',codes)
            status['stage']='plot';write_json(output/'status.json',status)
            plot_diagnostics(output)
            log({'complete':report})
            status.update(status='completed',stage='done')
        except BaseException as error:
            status.update(status='failed',error=repr(error))
            traceback.print_exc()
            raise
        finally:
            status['elapsed_seconds']=time.monotonic()-started
            write_json(output/'status.json',status)
            sys.stdout,sys.stderr = old_stdout,old_stderr


if __name__=='__main__':
    main()
