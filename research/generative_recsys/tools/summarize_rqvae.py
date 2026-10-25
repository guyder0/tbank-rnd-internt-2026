"""Summarize completed and incomplete RQ-VAE experiments, including older runs.

Usage: baseline/.venv/bin/python -m research.generative_recsys.tools.summarize_rqvae
"""
import argparse
import csv
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]/'artifacts/rqvae_only')
    parser.add_argument('--output',type=Path,help='Optional CSV file')
    args = parser.parse_args()
    rows = []
    for path in sorted(args.root.glob('*/status.json')):
        run = path.parent
        status,config = read(path),read(run/'config.json')
        diagnostics = read(run/'tokenizer/diagnostics.json')
        last = diagnostics[-1] if diagnostics else {}
        rows.append({'run':run.name,'status':status.get('status'),'smoke':status.get('smoke'),
            'limited':bool(config.get('rqvae_max_steps')),
            'mode':config.get('rqvae_optimization','joint'),
            'affine':config.get('rqvae_affine',False),
            'optimizer':config.get('rqvae_optimizer'),'lr':config.get('rqvae_learning_rate'),
            'epochs_done':last.get('epoch'),'epochs_requested':config.get('rqvae_epochs'),
            'seed':config.get('seed'),'batch_size':config.get('rqvae_batch_size'),
            'reconstruction':last.get('catalog_losses',{}).get('reconstruction'),
            'gain_vs_mean':last.get('reconstruction_gain_vs_mean'),
            'used_codes':'/'.join(str(v['used']) for v in last.get('usage',[])),
            'input_sha256':read(run/'input_manifest.json').get('sha256'),
            'seconds':status.get('elapsed_seconds'),'device':status.get('device')})
    print('run | status | mode | affine | lr | epochs | gain_vs_mean | codes | limited')
    for row in rows:
        gain = row['gain_vs_mean']
        gain_text = f'{gain:.2%}' if gain is not None else 'n/a'
        print(f"{row['run']} | {row['status']} | {row['mode']} | {row['affine']} | {row['lr']} | "
              f"{row['epochs_done']}/{row['epochs_requested']} | "
              f"{gain_text} | {row['used_codes']} | {row['limited']}")
    if args.output and rows:
        with args.output.open('w',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
        print('CSV:',args.output)


if __name__=='__main__':
    main()
