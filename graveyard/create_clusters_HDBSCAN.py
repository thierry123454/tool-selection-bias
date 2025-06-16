# pip install umap-learn hdbscan

import numpy as np
import pickle
import json
from collections import defaultdict
import umap
import hdbscan

# load pre-computed embeddings + records
embeddings = np.load("embeddings_combined_openai.npy")

with open("api_metadata.json") as f:
    data = json.load(f)

#  reduce dimensionality with UMAP for speed & noise-reduction
reducer = umap.UMAP(
    n_neighbors=15,
    n_components=64,
    metric="cosine",
    random_state=42,
)
emb_reduced = reducer.fit_transform(embeddings)

# run HDBSCAN on the lower-dim space
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=20,
    metric="euclidean",
    cluster_selection_method="eom",
)
labels = clusterer.fit_predict(emb_reduced)

np.save("labels_hdbscan.npy", labels)

# collect into buckets
buckets = defaultdict(list)
for idx, lbl in enumerate(labels):
    buckets[int(lbl)].append(idx)

with open("buckets_hdbscan.pkl", "wb") as out:
    pickle.dump(buckets, out)

print("Found", len([l for l in set(labels) if l >= 0 ]), "clusters (+ noise)")