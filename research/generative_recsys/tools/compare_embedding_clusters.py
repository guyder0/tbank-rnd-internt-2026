"""Compare HDBSCAN on aligned saved item embeddings, without recommendation labels.

Usage: python -m research.generative_recsys.tools.compare_embedding_clusters RUN_A RUN_B
Uses the same seeded item sample, raw Euclidean and unwhitened PCA-32 spaces.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from research.generative_recsys.tools.diagnose_clusters import geometry, occupancy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs=2, type=Path)
    args = parser.parse_args()
    manifests = [json.loads((run/'prepare_manifest.json').read_text()) for run in args.runs]
    ids = [pd.read_csv(run/'semantic_ids.csv').item_id.to_numpy() for run in args.runs]
    if not np.array_equal(ids[0], ids[1]):
        raise ValueError('Item identities or order differ')
    if manifests[0]['texts_sha256'] != manifests[1]['texts_sha256']:
        raise ValueError('Item texts differ')
    sample = np.random.default_rng(42).choice(len(ids[0]), min(3000,len(ids[0])), replace=False)
    output = args.runs[0]/'cluster_comparison'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output.mkdir(parents=True,exist_ok=False)
    report = {'settings':{'seed':42,'sample_size':len(sample),'metric':'euclidean',
        'min_samples':10,'cluster_selection_method':'eom','allow_single_cluster':True,
        'pca_dimensions':32,'pca_whiten':False,'pca_fit':'full catalog'},
        'catalog_size':len(ids[0]),'texts_sha256':manifests[0]['texts_sha256'],'runs':[]}
    assignments = pd.DataFrame({'item_id':ids[0][sample]})
    started = time.monotonic()
    with threadpool_limits(limits=4):
        for run, manifest in zip(args.runs,manifests):
            path = run/'item_embeddings.npy'
            x = np.load(path)
            if len(x) != len(ids[0]) or not np.isfinite(x).all():
                raise ValueError('Invalid embedding matrix')
            backend = manifest['embedding']['backend']
            info = {'run':str(run),'backend':backend,
                'embeddings_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'geometry':geometry(x,42),'hdbscan':[]}
            pca = PCA(n_components=min(32,x.shape[1]),svd_solver='randomized',random_state=42)
            reduced = pca.fit_transform(x)
            info['pca_variance_retained'] = float(pca.explained_variance_ratio_.sum())
            for space,values in [('pca32',reduced[sample]),('original',x[sample])]:
                for minimum in [15,50,150]:
                    labels = HDBSCAN(min_cluster_size=minimum,min_samples=10,
                        cluster_selection_method='eom',allow_single_cluster=True,n_jobs=4,
                        copy=True).fit_predict(values)
                    row = {'space':space,'min_cluster_size':minimum,**occupancy(labels)}
                    info['hdbscan'].append(row)
                    assignments[f'{backend}_{space}_{minimum}'] = labels
                    print(backend,row,flush=True)
            report['runs'].append(info)
            (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    report['elapsed_seconds'] = time.monotonic()-started
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    assignments.to_csv(output/'assignments.csv',index=False)
    print('Report:',output/'report.json',flush=True)


if __name__ == '__main__':
    main()
