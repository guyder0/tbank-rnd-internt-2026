"""CPU-only geometry diagnostics of saved embeddings; never changes models.

Run: baseline/.venv/bin/python -u -m research.generative_recsys.tools.diagnose_clusters RUN_DIR
K-means occupancy is not an estimate of an intrinsic cluster count. HDBSCAN
results are density/parameter dependent; report noise and PCA preprocessing.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits
import torch

from research.generative_recsys.src.rqvae import RQVAE


def occupancy(labels):
    _, counts = np.unique(labels[labels >= 0], return_counts=True)
    p = counts/counts.sum() if len(counts) else np.array([])
    return {'clusters':len(counts), 'noise_fraction':float(np.mean(labels < 0)),
            'min_size':int(counts.min()) if len(counts) else 0,
            'max_size':int(counts.max()) if len(counts) else 0,
            'effective_clusters':float(np.exp(-(p*np.log(p)).sum())) if len(p) else 0.}


def geometry(x, seed):
    rng = np.random.default_rng(seed)
    first = rng.integers(len(x), size=20000)
    second = (first+rng.integers(1,len(x),size=len(first))) % len(x)
    unit = x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
    cosine = (unit[first]*unit[second]).sum(1)
    l2 = np.linalg.norm(x[first]-x[second],axis=1)
    variance = float(np.mean(np.sum((x-x.mean(0))**2,axis=1)))
    return {'shape':list(x.shape), 'unique_vectors':len(np.unique(x,axis=0)),
        'centered_variance':variance,
        'random_pair_cosine_quantiles':np.quantile(cosine,[.05,.5,.95]).tolist(),
        'random_pair_l2_quantiles':np.quantile(l2,[.05,.5,.95]).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    output = args.run/'cluster_diagnostics'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output.mkdir(parents=True,exist_ok=False)
    source = args.run/'item_embeddings.npy'
    x = np.load(source)
    checkpoint = torch.load(args.run/'tokenizer/model.pt',map_location='cpu',weights_only=True)
    model = RQVAE(**checkpoint['settings'])
    model.load_state_dict(checkpoint['state_dict']); model.eval()
    with torch.inference_mode():
        z = torch.cat([model.encoder(torch.from_numpy(x[i:i+1024])) for i in range(0,len(x),1024)]).numpy()
        actual = model.quantize(torch.from_numpy(z))[1][:,0].numpy()
    report = {'seed':42,'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'rqvae_sha256':hashlib.sha256((args.run/'tokenizer/model.pt').read_bytes()).hexdigest(),
        'settings':{'kmeans_n_init':3,'kmeans_max_iter':100,'silhouette_sample':1000,
                    'hdbscan_sample':3000,'hdbscan_min_samples':10,'pca_whiten':False},
        'rqvae_level_one':occupancy(actual),'spaces':{}}
    rng = np.random.default_rng(42)
    subset = rng.choice(len(x),min(3000,len(x)),replace=False)
    start = time.monotonic()
    with threadpool_limits(limits=4):
        for name, values in [('sentence_t5',x),('rqvae_latent',z)]:
            info = geometry(values,42)
            info['kmeans'] = []
            for k in [2,8,27,64]:
                estimator = KMeans(n_clusters=k,n_init=3,max_iter=100,random_state=42)
                labels = estimator.fit_predict(values)
                row = {'requested_k':k,**occupancy(labels),
                    'variance_explained':1-float(estimator.inertia_)/(len(values)*info['centered_variance']),
                    'silhouette_euclidean':float(silhouette_score(values,labels,sample_size=1000,random_state=42))}
                info['kmeans'].append(row)
                print(name,'kmeans',row,flush=True)
            pca = PCA(n_components=min(32,values.shape[1]),svd_solver='randomized',random_state=42)
            reduced = pca.fit_transform(values)
            info['pca_variance_retained'] = float(pca.explained_variance_ratio_.sum())
            info['hdbscan_pca_sample'] = []
            for minimum in [15,50,150]:
                labels = HDBSCAN(min_cluster_size=minimum,min_samples=10,
                    cluster_selection_method='eom',allow_single_cluster=True,n_jobs=4,copy=True).fit_predict(reduced[subset])
                row = {'min_cluster_size':minimum,**occupancy(labels)}
                info['hdbscan_pca_sample'].append(row)
                print(name,'hdbscan',row,flush=True)
            report['spaces'][name] = info
            (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    report['elapsed_seconds'] = time.monotonic()-start
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print('Report:',output/'report.json',flush=True)


if __name__ == '__main__':
    main()
