"""Logged GPU training; --smoke runs a small complete CPU pipeline."""
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from baseline.pipeline import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true', help='300 users, 3 iterations, CPU')
    parser.add_argument('--device', choices=['GPU', 'CPU'], default='GPU')
    parser.add_argument('--devices', default='0', help='CatBoost GPU IDs')
    parser.add_argument('--run-dir', type=Path)
    args = parser.parse_args()
    expected = ROOT/'baseline/.venv'
    if Path(sys.prefix).resolve() != expected.resolve():
        parser.error('Use baseline/.venv/bin/python; other Python environments are not supported')
    os.chdir(ROOT)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = (args.run_dir or ROOT/'baseline/artifacts/runs'/stamp).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((ROOT/'baseline/configs/strong.json').read_text())
    config.update(extended_aggregates=True, task_type='CPU' if args.smoke else args.device,
                  devices=args.devices, gpu_ram_part=0.75,
                  graph_cache_directory=str(ROOT/'baseline/artifacts/strong'),
                  reference_model=str(ROOT/'baseline/artifacts/strong/expanded/model.cbm'))
    if args.smoke:
        config.update(smoke_users=300, iterations=3, thread_count=2)
    config_path = output/'config.json'
    config_path.write_text(json.dumps(config, indent=2))
    source = output/'source'; source.mkdir()
    for filename in ['pipeline.py', 'aggregates.py', 'strong.py', 'run_training.py']:
        shutil.copy2(ROOT/'baseline'/filename, source/filename)
    status = {'status': 'running', 'smoke': args.smoke, 'task_type': config['task_type'],
              'started_utc': stamp, 'python': sys.version, 'executable': sys.executable,
              'platform': platform.platform(), 'pid': os.getpid(),
              'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}}
    started = time.monotonic()
    with (output/'run.log').open('w', buffering=1) as logfile:
        def log(message):
            line = str(message).rstrip('\n')
            print(line, flush=True); logfile.write(line+'\n')
        def save_status():
            (output/'status.json').write_text(json.dumps(status, indent=2))
        def run(command):
            log('$ '+' '.join(str(x) for x in command))
            process = subprocess.Popen([str(x) for x in command], stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                for line in process.stdout:
                    log(line)
                code = process.wait()
            except BaseException:
                process.terminate(); process.wait(); raise
            if code:
                raise RuntimeError(f'Command exited with code {code}: {command}')
        log(f'Run directory: {output}')
        save_status()
        try:
            status['stage'] = 'preflight'; save_status()
            for required in ['manifest.json', 'sample_users.csv', 'model.cbm']:
                if not (ROOT/'baseline/artifacts/default'/required).exists():
                    raise FileNotFoundError(f'Missing baseline/artifacts/default/{required}')
            if not Path(config['reference_model']).exists():
                raise FileNotFoundError(config['reference_model'])
            run([sys.executable, '-c', "from importlib.metadata import distributions; print('\\n'.join(sorted(d.metadata['Name']+'=='+d.version for d in distributions())))"])
            # Execute the actual backend before the expensive data preparation.
            check = ("from catboost import CatBoostRanker; "
                     "m=CatBoostRanker(iterations=2,depth=2,verbose=False,allow_writing_files=False,"
                     f"task_type={config['task_type']!r},"
                     +(f"devices={args.devices!r},gpu_ram_part=0.75," if config['task_type']=='GPU' else 'thread_count=2,')
                     +"); m.fit([[0],[1],[0],[1]],[0,1,0,1],group_id=[0,0,1,1]); print('Backend preflight OK')")
            run([sys.executable, '-u', '-c', check])
            for stage in ['prepare', 'train']:
                status['stage'] = stage; save_status()
                stage_start = time.monotonic()
                run([sys.executable, '-u', '-m', 'baseline.strong', stage,
                     '--variant', 'expanded', '--config', config_path, '--output', output])
                status[stage+'_seconds'] = time.monotonic()-stage_start
            status.update(status='completed', stage='done')
            log('Completed. Metrics: expanded/metrics.json; model: expanded/model.cbm')
            log('Only validation evaluated; historical holdout was not evaluated again.')
        except BaseException as error:
            status.update(status='failed', error=repr(error))
            log(traceback.format_exc())
            raise
        finally:
            status['elapsed_seconds'] = time.monotonic()-started
            save_status()


if __name__ == '__main__':
    main()
