import numpy as np
import json
from collections import defaultdict, Counter

with open("api_metadata.json", "r", encoding="utf-8") as f:
    data = json.load(f)

records = []
for cat, tools in data.items():
    for tool, info in tools.items():
        tool_desc = info.get("tool_desc", "").strip()
        for name, desc in info["apis"]:
            # bake in the tool description
            text = (
                f"{tool}: {tool_desc} | "
                f"{name}: {desc}"
            )
            records.append({"tool": tool, "text": text})

labels = np.load("labels_hdbscan.npy")

clusters = defaultdict(list)
for rec, lbl in zip(records, labels):
    if lbl == -1:
        continue
    clusters[lbl].append(rec)

sizes = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
print("Top 5 largest clusters:")
for lbl, recs in sizes[:5]:
    print(f"  Cluster {lbl}: {len(recs)} items")

print("\nBottom 5 smallest clusters:")
for lbl, recs in sizes[-5:]:
    print(f"  Cluster {lbl}: {len(recs)} items")

print("\nSample contents of top 5 clusters:")
for lbl, recs in sizes[35:40]:
    print(f"\nCluster {lbl} (size={len(recs)}):")
    print("  Example APIs:")
    for r in recs[:20]:
        print("   -  ", r["text"])


# estimate total NLI ops = sum of squares of cluster sizes
total_ops = sum(len(recs)**2 for recs in clusters.values())
print(f"\nEstimated NLI operations (sum of squares): {total_ops:,}")

# if you want the exact number of pairwise comparisons:
pairwise_ops = sum(len(recs)*(len(recs)-1)//2 for recs in clusters.values())
print(f"Estimated pairwise comparisons:        {pairwise_ops:,}")