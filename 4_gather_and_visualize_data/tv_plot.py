import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# ─── CONFIG ────────────────────────────────────────────────────────────
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
CLUSTERS_JSON = "../2_generate_clusters_and_refine/duplicate_api_clusters.json"

# SIZE
STATS_PATHS = {
    "1.7": ["api_selection_stats_qwen-1.7b.json", "api_selection_stats_qwen-1.7b-2.json", "api_selection_stats_qwen-1.7b-3.json"],
    "4":   ["api_selection_stats_qwen-4b.json", "api_selection_stats_qwen-4b-2.json", "api_selection_stats_qwen-4b-3.json"],
    "8":   ["api_selection_stats_qwen-8b.json", "api_selection_stats_qwen-8b-2.json", "api_selection_stats_qwen-8b-3.json"] 
}
BASE = "tv_by_size"
OUTPUT_PDF     = BASE + ".pdf"
OUTPUT_PNG     = BASE + ".png"
X_AXIS = "Model size (B parameters)"
PARAMETER = "Model Scale"

# TEMPERATURE
# STATS_PATHS = {
#     "0": ["api_selection_stats_chatgpt-temp-0.json", "api_selection_stats_chatgpt-temp-0-1.json", "api_selection_stats_chatgpt-temp-0-2.json"],
#     "0.5": ["api_selection_stats_chatgpt_base.json", "api_selection_stats_chatgpt-temp-0.5-2.json", "api_selection_stats_chatgpt-temp-0.5-3.json"],
#     "1":  ["api_selection_stats_chatgpt-temp-1.json", "api_selection_stats_chatgpt-temp-1-2.json", "api_selection_stats_chatgpt-temp-1-3.json"]
# }
# BASE = "tv_by_temp"
# OUTPUT_PDF     = BASE + ".pdf"
# OUTPUT_PNG     = BASE + ".png"
# X_AXIS = "Temperature"
# PARAMETER = "Temperature"

# TOP-P
# STATS_PATHS = {
#     "0.7": "api_selection_stats_chatgpt-top-p-0.7.json",
#     "0.9": "api_selection_stats_chatgpt-top-p-0.9.json",
#     "1":  "api_selection_stats_chatgpt_base.json"
# }
# BASE = "tv_by_top_p"
# OUTPUT_PDF     = BASE + ".pdf"
# OUTPUT_PNG     = BASE + ".png"
# X_AXIS = "Top-$p$"
# PARAMETER = "Top-$p$"
# ────────────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# load clusters once
clusters = load_json(CLUSTERS_JSON)

def average_tv_for_stats(stats):
    """Given one stats list, compute the mean combined TV across all clusters."""
    # counts for this single run
    counts_api = defaultdict(lambda: defaultdict(int))
    counts_pos = defaultdict(lambda: defaultdict(int))
    for _, cid, pos_cl, pos_list in stats:
        counts_api[cid][pos_cl] += 1
        counts_pos[cid][pos_list] += 1

    # compute rates
    rates_api = {}
    rates_pos = {}
    for cid in counts_api:
        total = sum(counts_api[cid].values())
        rates_api[cid] = {i: counts_api[cid][i]/total for i in counts_api[cid]} if total else {}
    for cid in counts_pos:
        total = sum(counts_pos[cid].values())
        rates_pos[cid] = {i: counts_pos[cid][i]/total for i in counts_pos[cid]} if total else {}

    # compute combined TV per cluster
    tv_combined = []
    for cid, cluster in enumerate(clusters, start=1):
        K = len(cluster)
        uniform = 1.0 / K
        # API‐level TV
        p_api = rates_api.get(cid, {})
        tv_api = 0.5 * sum(abs(p_api.get(i, 0.0) - uniform) for i in range(1, K+1))
        # pos‐level TV
        p_pos = rates_pos.get(cid, {})
        tv_pos = 0.5 * sum(abs(p_pos.get(i, 0.0) - uniform) for i in range(1, K+1))
        tv_combined.append((tv_api + tv_pos) / 2)
    return np.mean(tv_combined)

# compute, for each temperature, the list of average TVs across runs
avg_runs = {}
for temp, paths in STATS_PATHS.items():
    runs = []
    for p in paths:
        stats = load_json(p)
        runs.append(average_tv_for_stats(stats))
    avg_runs[temp] = runs

# now compute mean and std per temperature
temps = sorted(avg_runs, key=lambda t: float(t))
mean_vals = [np.mean(avg_runs[t]) for t in temps]
std_vals  = [np.std (avg_runs[t], ddof=1) for t in temps]

# plot with error bars
plt.figure(figsize=(6,4))
x = [float(t) for t in temps]
plt.errorbar(x, mean_vals, yerr=std_vals, marker='o', capsize=5, linestyle='-')
plt.xlabel(X_AXIS)
plt.ylabel("$\delta_{\mathrm{model}}$")
plt.title(f"Tool-Selection Bias vs. {PARAMETER}")
plt.grid(True)
plt.tight_layout()

# save & show
plt.savefig(OUTPUT_PDF, format="pdf", transparent=True)
plt.savefig(OUTPUT_PNG, format="png", transparent=True)
print(f"Saved plots to {OUTPUT_PDF} and {OUTPUT_PNG}")

plt.show()