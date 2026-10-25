"""Copy the already trained best GPU weights into a mounted data directory."""
import argparse
import json
from pathlib import Path
import tempfile
import shutil

from catboost import CatBoostRanker
from baseline_docker.app import load_config, sha256, source_hashes, write_json, directory_lock

ROOT=Path(__file__).resolve().parents[1]
BEST=ROOT/'baseline/artifacts/refined/full_v1/combined_seed44/expanded'


def export_best(directory):
    directory=Path(directory).resolve()
    if not directory.is_dir():
        raise ValueError('The mounted data directory must already exist')
    with directory_lock(directory):
        if (directory/'model.cbm').exists() or (directory/'model.json').exists():
            raise ValueError('Existing model artifacts would be overwritten; choose another directory')
        source=BEST/'model.cbm'
        metrics=json.loads((BEST/'metrics.json').read_text())
        config=load_config()
        model=CatBoostRanker().load_model(str(source))
        if model.feature_names_!=config['feature_columns'] or metrics['config']['seed']!=44:
            raise ValueError('Source model is not the expected combined seed44')
        metadata={'schema_version':1,'origin':'pretrained_combined_seed44','config':config,
                  # Labels used in validation selection cover [Aug 9, Aug 16).
                  'trained_before':'2021-08-16','source_sha256':source_hashes(),
                  'model_sha256':sha256(source),'training_backend':'GPU',
                  'training':{'trees':model.tree_count_,'validation_ndcg@10':metrics['validation']['catboost']['ndcg@10'],
                              'limitation':'pretrained on original KION development users; not trained from mounted train.csv'},
                  'input_sha256':{}}
        with tempfile.TemporaryDirectory(prefix='.model-pending-',dir=directory) as temporary:
            pending=Path(temporary)/'model.cbm';shutil.copy2(source,pending)
            write_json(directory/'model.json',metadata)
            pending.replace(directory/'model.cbm')
    print(f'Exported original combined seed44 model to {directory}/model.cbm')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    export_best(parser.parse_args().data_dir)
