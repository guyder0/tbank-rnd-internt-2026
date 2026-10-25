"""Attach an already trained RQ-VAE to the exact original recommendation splits.

No text encoder or tokenizer training. CPU encode verifies saved raw codes.
"""
import argparse
import json
from pathlib import Path
import pickle
import shutil
import time

import numpy as np
import pandas as pd
import torch

from research.generative_recsys.run import HOME, digest, write_json
from research.generative_recsys.src.rqvae import RQVAE, encode_all
from research.generative_recsys.src.tokenizer import finalize_codes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rqvae-run',type=Path,default=HOME/'artifacts/rqvae_only/20261025T185949_751066Z_alt_affine_lr01_ep1000')
    parser.add_argument('--output',type=Path,default=HOME/'artifacts/kion_e5_affine_frozen')
    args = parser.parse_args()
    started = time.monotonic()
    torch.set_num_threads(4)
    run, output = args.rqvae_run.resolve(),args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Output already exists; choose a new --output: {output}')
    status = json.loads((run/'status.json').read_text())
    if status['status']!='completed' or status.get('smoke'):
        raise ValueError('A completed non-smoke RQ-VAE run is required')
    source_info = json.loads((run/'input_manifest.json').read_text())
    if source_info['sampled']:
        raise ValueError('Sampled embeddings cannot replace the full catalog')
    source = Path(source_info['source']).parent
    manifest = json.loads((source/'prepare_manifest.json').read_text())
    embeddings_hash = digest(source/'item_embeddings.npy')
    if embeddings_hash!=source_info['sha256'] or embeddings_hash!=manifest['embeddings_sha256']:
        raise ValueError('Embedding checksum mismatch')
    if digest(source/'prepared.pkl')!=manifest['prepared_sha256']:
        raise ValueError('Source prepared checksum mismatch')
    with (source/'prepared.pkl').open('rb') as stream:
        prepared = pickle.load(stream)
    embeddings = np.load(source/'item_embeddings.npy',allow_pickle=False)
    checkpoint_path = run/'tokenizer/model.pt'
    checkpoint = torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    if checkpoint['config'].get('rqvae_max_steps'):
        raise ValueError('Limited-step RQ-VAE run cannot be used as a full model')
    config = prepared['config'].copy()
    config.update({key:value for key,value in checkpoint['config'].items() if key.startswith('rqvae_')})
    config.update(tokenizer='rqvae',seed=checkpoint['config']['seed'])
    model = RQVAE(**checkpoint['settings'])
    model.load_state_dict(checkpoint['state_dict']);model.eval()
    codes,_ = encode_all(model,embeddings,1024,'cpu')
    saved = np.load(run/'codes.npy',allow_pickle=False)
    if not np.array_equal(codes,saved):
        raise ValueError('Checkpoint encoding differs from saved codes')
    ids = prepared['codebook']['item_ids']
    if len(ids)!=len(codes) or codes.shape[1]!=config['code_levels']:
        raise ValueError('Catalog/code shape mismatch')
    source_ids = pd.read_csv(source/'semantic_ids.csv').item_id.to_numpy()
    if not np.array_equal(ids,source_ids):
        raise ValueError('Catalog item order mismatch')
    report = json.loads((run/'tokenizer/metrics.json').read_text())
    books = [book.detach().numpy().copy() for book in model.effective_codebooks()]
    prepared['codebook'] = finalize_codes(ids,codes,books,config,report)
    prepared['config'] = config
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'config.json',config)
    with (output/'prepared.pkl').open('wb') as stream:
        pickle.dump(prepared,stream,protocol=5)
    shutil.copy2(source/'item_embeddings.npy',output/'item_embeddings.npy')
    (output/'tokenizer').mkdir()
    shutil.copy2(checkpoint_path,output/'tokenizer/model.pt')
    shutil.copy2(run/'tokenizer/metrics.json',output/'tokenizer/metrics.json')
    columns = {f'code_{level}':prepared['codebook']['codes'][:,level]
               for level in range(prepared['codebook']['codes'].shape[1])}
    pd.DataFrame({'item_id':ids,**columns}).to_csv(output/'semantic_ids.csv',index=False)
    write_json(output/'prepare_manifest.json',{**manifest,'config':config,
        'prepared_sha256':digest(output/'prepared.pkl'),
        'diagnostics':prepared['codebook']['diagnostics'],'prepare_seconds':time.monotonic()-started,
        'frozen_rqvae':{'source_run':str(run),'checkpoint_sha256':digest(checkpoint_path),
            'codes_sha256':digest(run/'codes.npy'),'original_prepared':str(source),
            'checkpoint_codes_match':True,'retrained':False}})
    shutil.copy2(Path(__file__),output/'prepare_source.py')
    print('Prepared frozen RQ-VAE:',output,flush=True)
    print('Diagnostics:',prepared['codebook']['diagnostics'],flush=True)


if __name__=='__main__':
    main()
