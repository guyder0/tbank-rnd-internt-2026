"""Export the exact strongest measured generator and frozen SID catalog."""
import argparse
import json
from pathlib import Path
import pickle

import torch

from generative_docker.app import load_config, load_model, save_bundle
from generative_docker.io import directory_lock, sha256

ROOT=Path(__file__).resolve().parents[1]
BEST=ROOT/'research/generative_recsys/artifacts/20261025T170957_093408Z'


def export_best(directory):
    directory=Path(directory).resolve()
    if not directory.is_dir():raise ValueError('Data directory must already exist')
    with directory_lock(directory):
        manifest=json.loads((BEST/'prepare_manifest.json').read_text())
        if sha256(BEST/'prepared.pkl')!=manifest['prepared_sha256']:raise ValueError('Source cache checksum mismatch')
        with (BEST/'prepared.pkl').open('rb') as stream:prepared=pickle.load(stream)
        checkpoint=torch.load(BEST/'model.pt',map_location='cpu',weights_only=True)
        recipe=load_config()
        for key,value in checkpoint['config'].items():
            if recipe[key]!=value:raise ValueError(f'Winning recipe differs in {key}')
        checkpoint['config']=recipe
        codebook=prepared['codebook']
        bundle={'schema_version':1,'checkpoint':checkpoint,
                'item_ids':torch.as_tensor(codebook['item_ids']),'codes':torch.as_tensor(codebook['codes'])}
        load_model(bundle,torch.device('cpu'))
        metadata={'schema_version':1,'origin':'exported_best_tfidf_rqvae',
            'config':recipe,'trained_before':'2021-08-16','training_backend':'cuda',
            'source_model_sha256':sha256(BEST/'model.pt'),'source_prepared_sha256':manifest['prepared_sha256'],
            'input_sha256':{},'training':json.loads((BEST/'metrics.json').read_text()),
            'limitation':'Original KION development users and validation selection; not fitted to mounted train.csv'}
        save_bundle(bundle,directory,metadata)
    print(f'Exported exact best generator and SID catalog to {directory}/sid_model.pt')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    export_best(parser.parse_args().data_dir)
