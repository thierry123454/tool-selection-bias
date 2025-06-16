import json
from collections import Counter

# path to your cluster file
CLUSTERS_PATH = "duplicate_api_clusters.json"

# load the clusters
with open(CLUSTERS_PATH, "r", encoding="utf-8") as f:
    clusters = json.load(f)

# how many clusters?
n_clusters = len(clusters)

# compute each cluster’s size
sizes = [len(cluster) for cluster in clusters]

# summary statistics
min_size = min(sizes) if sizes else 0
max_size = max(sizes) if sizes else 0
avg_size = sum(sizes) / n_clusters if n_clusters else 0

# distribution of sizes
size_counts = Counter(sizes)

print(f"Total clusters: {n_clusters}")
print(f"Cluster size — min: {min_size}, max: {max_size}, avg: {avg_size:.2f}\n")

print("Cluster size distribution:")
for size, count in sorted(size_counts.items()):
    print(f"  size {size:2d}: {count} cluster{'s' if count>1 else ''}")

