import json
from collections import defaultdict
import matplotlib.pyplot as plt

# ─── CONFIG ────────────────────────────────────────────────────────────
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# SIZE
# STATS_PATHS = {
#     "1.7B": "api_selection_stats_qwen-1.7b.json",
#     "4B":   "api_selection_stats_qwen-4b.json",
#     "8B":   "api_selection_stats_qwen-8b.json",
# }
# CLUSTERS_JSON = "../2_generate_clusters_and_refine/duplicate_api_clusters.json"
# BASE = "tv_by_size"
# OUTPUT_PDF     = BASE + ".pdf"
# OUTPUT_PNG     = BASE + ".png"
# X_AXIS = "Model size (B parameters)"
# Y_AXIS = "Avg. combined TV distance"
# PARAMETER = "Model Scale"

# TEMPERATURE
# STATS_PATHS = {
#     "0": "api_selection_stats_chatgpt-temp-0.json",
#     "0.5":  "api_selection_stats_chatgpt_base.json",
#     "1":  "api_selection_stats_chatgpt-temp-1.json"
# }
# CLUSTERS_JSON = "../2_generate_clusters_and_refine/duplicate_api_clusters.json"
# BASE = "tv_by_temp"
# OUTPUT_PDF     = BASE + ".pdf"
# OUTPUT_PNG     = BASE + ".png"
# X_AXIS = "Temperature"
# Y_AXIS = "Avg. combined TV distance"
# PARAMETER = "Temperature"

# TOP-P
STATS_PATHS = {
    "0.7": "api_selection_stats_chatgpt-top-p-0.7.json",
    "0.9": "api_selection_stats_chatgpt-top-p-0.9.json",
    "1":  "api_selection_stats_chatgpt_base.json"
}
CLUSTERS_JSON = "../2_generate_clusters_and_refine/duplicate_api_clusters.json"
BASE = "tv_by_top_p"
OUTPUT_PDF     = BASE + ".pdf"
OUTPUT_PNG     = BASE + ".png"
X_AXIS = "Top-$p$"
Y_AXIS = "Avg. combined TV distance"
PARAMETER = "Top-$p$"
# ────────────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# Load clusters & model‐specific stats
clusters    = load_json(CLUSTERS_JSON)
model_stats = {name: load_json(path) for name, path in STATS_PATHS.items()}

# Build counts for (a) pos_in_cluster and (b) pos_in_list
counts_api = {m: defaultdict(lambda: defaultdict(int)) for m in model_stats}
counts_pos = {m: defaultdict(lambda: defaultdict(int)) for m in model_stats}

for model, stats in model_stats.items():
    for _, cid, pos_cluster, pos_list in stats:
        counts_api[model][cid][pos_cluster] += 1
        counts_pos[model][cid][pos_list]    += 1

# Compute selection‐rate p_i = count_i / total_counts
rates_api = {m: {} for m in counts_api}
rates_pos = {m: {} for m in counts_pos}

for m in model_stats:
    for cid, pd in counts_api[m].items():
        total = sum(pd.values())
        rates_api[m][cid] = {i: pd[i]/total for i in pd} if total else {}
    for cid, pd in counts_pos[m].items():
        total = sum(pd.values())
        rates_pos[m][cid] = {i: pd[i]/total for i in pd} if total else {}

# Compute TV‐distances & combined metric per cluster
tv_api_by_model      = {m: [] for m in STATS_PATHS}
tv_pos_by_model      = {m: [] for m in STATS_PATHS}
tv_combined_by_model = {m: [] for m in STATS_PATHS}

for cid, cluster in enumerate(clusters, start=1):
    K = len(cluster)
    uniform = 1.0 / K
    for m in STATS_PATHS:
        p_api = rates_api[m].get(cid, {})
        p_pos = rates_pos[m].get(cid, {})
        tv_api = 0.5 * sum(abs(p_api.get(i, 0.0) - uniform) for i in range(1, K+1))
        tv_pos = 0.5 * sum(abs(p_pos.get(i, 0.0) - uniform) for i in range(1, K+1))
        tv_combined = (tv_api + tv_pos) / 2

        tv_api_by_model[m].append(tv_api)
        tv_pos_by_model[m].append(tv_pos)
        tv_combined_by_model[m].append(tv_combined)

# Compute average TV per model
avg_combined = {
    m: sum(tv_combined_by_model[m]) / len(tv_combined_by_model[m])
    for m in STATS_PATHS
}

# Plotting
model_sizes = [float(size.replace("B","")) for size in STATS_PATHS]  # [1.7, 4, 8]
combined_vals = [avg_combined[m] for m in STATS_PATHS]

plt.figure()
plt.plot(model_sizes, combined_vals, marker='o')
plt.xlabel(X_AXIS)
plt.ylabel(Y_AXIS)
plt.title(f"Tool-Selection Bias vs. {PARAMETER}")
plt.grid(True)
plt.tight_layout()

# save & show
plt.savefig(OUTPUT_PDF, format="pdf", transparent=True)
print(f"Saved chart grid to {OUTPUT_PDF}")

plt.savefig(OUTPUT_PNG, format="png", transparent=True)
print(f"Saved chart grid to {OUTPUT_PNG}")

plt.show()