"""Stage only the runtime files needed by the Docker image; no data or models."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
FILES = ['baseline/__init__.py', 'baseline/pipeline.py', 'baseline/aggregates.py',
         'baseline/refined.py', 'baseline_docker/__init__.py', 'baseline_docker/app.py',
         'baseline_docker/features.py', 'baseline_docker/default_config.json',
         'baseline_docker/requirements.txt']


def stage(destination=None):
    destination = Path(destination) if destination else ROOT/'baseline_docker/.build-context'
    if destination.exists():
        # Remove only the dedicated default generated directory, never arbitrary paths.
        if destination.resolve() != (ROOT/'baseline_docker/.build-context').resolve():
            raise ValueError('Custom staging directory must not exist')
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    hashes = {}
    for name in FILES:
        source = ROOT/name
        target = destination/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    shutil.copy2(ROOT/'baseline_docker/Dockerfile', destination/'Dockerfile')
    (destination/'baseline_docker/source_manifest.json').write_text(json.dumps(hashes, indent=2))
    print(destination)
    return destination


if __name__ == '__main__':
    stage()
