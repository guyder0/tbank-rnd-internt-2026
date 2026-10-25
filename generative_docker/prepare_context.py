"""Small allowlisted Docker context, excluding datasets, models and experiments."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
FILES=['baseline/pipeline.py','generative_docker/__init__.py','generative_docker/app.py',
       'generative_docker/io.py','generative_docker/default_config.json','generative_docker/requirements.lock.txt']
FILES += ['research/generative_recsys/src/'+name+'.py' for name in ['data','decoding','metrics','model','tokenizer','training','rqvae']]


def stage(destination=None):
    destination=Path(destination) if destination else ROOT/'generative_docker/.build-context'
    if destination.exists():
        if destination.resolve()!=(ROOT/'generative_docker/.build-context').resolve():raise ValueError('Custom destination must not exist')
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    hashes={}
    for name in FILES:
        source=ROOT/name;target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target);hashes[name]=hashlib.sha256(source.read_bytes()).hexdigest()
    shutil.copy2(ROOT/'generative_docker/Dockerfile',destination/'Dockerfile')
    (destination/'generative_docker/source_manifest.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(destination)
    return destination


if __name__=='__main__':stage()
