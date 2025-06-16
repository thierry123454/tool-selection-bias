from sklearn.cluster import MiniBatchKMeans
import numpy as np
from collections import defaultdict
import json
import pickle

with open("api_metadata.json") as f:
    data = json.load(f)

embeddings = np.load("embeddings_no_tool_openai.npy")

# 2) cluster with MiniBatchKMeans
n_clusters = 500
kmeans = MiniBatchKMeans(
    n_clusters=n_clusters,
    batch_size=1024,  # try 512–4096
    max_iter=100,
    n_init=3,         # a few inits for stability
    random_state=42
)
labels = kmeans.fit_predict(embeddings)

print(labels)

np.save("labels.npy", labels)

# group record indices by cluster
buckets = defaultdict(list)
for idx, lbl in enumerate(labels):
    if lbl >= 0:
        buckets[lbl].append(idx)

with open("buckets.pkl","wb") as out:
    pickle.dump(buckets, out)