import json
from collections import defaultdict

# ─── CONFIG ────────────────────────────────────────────────────────────
STATS_PATHS = {
    "ChatGPT":   "api_selection_stats_chatgpt.json",
    "Claude":    "api_selection_stats_claude.json",
    "Gemini":    "api_selection_stats_gemini.json",
    "DeepSeek":  "api_selection_stats_deepseek.json",
}
CLUSTERS_JSON = "2_generate_clusters_and_refine/duplicate_api_clusters.json"
# ────────────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# Load clusters & model‐specific stats
clusters    = load_json(CLUSTERS_JSON)
model_stats = {name: load_json(path) for name, path in STATS_PATHS.items()}

# Build counts[model][cluster_id][pos_in_cluster]
counts = {m: defaultdict(lambda: defaultdict(int)) for m in model_stats}
for model, stats in model_stats.items():
    for _, cid, pos, _ in stats:
        counts[model][cid][pos] += 1

# Compute selection‐rate p_i = count_i / total_counts for each model & cluster
rates = {m: {} for m in counts}

for m, cldict in counts.items():
    for cid, posdict in cldict.items():
        total = sum(posdict.values())
        if total > 0:
            rates[m][cid] = {pos: cnt/total for pos, cnt in posdict.items()}
        else:
            rates[m][cid] = {}

# Now compute TV distance to uniform for each (model, cluster)
print("TV distances to uniform by model & cluster:\n")
tv_by_model = {m: [] for m in STATS_PATHS}

for cid, cluster in enumerate(clusters, start=1):
    K = len(cluster)
    uniform = 1.0 / K
    print(f"Cluster {cid:2d} ({K} endpoints):")
    for m in STATS_PATHS:
        p = rates[m].get(cid, {})
        # ensure we include zero‐selections
        tv = 0.5 * sum(abs(p.get(i, 0.0) - uniform) for i in range(1, K+1))
        print(f"   {m:8s}: TV = {tv:.3f}")
        tv_by_model[m].append(tv)
    print()

avg_tv = {m: (sum(tvs) / len(tvs) if tvs else 0.0) for m, tvs in tv_by_model.items()}

print("Average TV distance to uniform, across all clusters:")
for m, atv in avg_tv.items():
    print(f"  {m:8s}: {atv:.3f}")