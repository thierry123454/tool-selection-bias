import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Setup LaTeX
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# ─── CONFIG ────────────────────────────────────────────────────────────
# STATS_PATHS = {
#     "ChatGPT 3.5":  "api_selection_stats_chatgpt_base.json",
#     "ChatGPT 4.1":  "api_selection_stats_chatgpt_4.json",
#     "Claude":  "api_selection_stats_claude.json",
#     "Gemini":  "api_selection_stats_gemini.json",
#     "DeepSeek":  "api_selection_stats_deepseek.json",
#     "Qwen (32B)":  "api_selection_stats_qwen-32b.json",
#     "ToolLLama":  "api_selection_stats_toolllama.json"
# }
STATS_PATHS = {
    "ChatGPT 3.5":  "api_selection_stats_chatgpt_base.json",
    "ChatGPT 4.1":  "api_selection_stats_chatgpt_4.json",
    "Gemini":  "api_selection_stats_gemini.json",
    "DeepSeek":  "api_selection_stats_deepseek.json",
    "Qwen":  "api_selection_stats_qwen-235b.json",
}
CLUSTERS_JSON = "../2_generate_clusters_and_refine/duplicate_api_clusters.json"

# map each cluster to its human-friendly tag
CLUSTER_NAMES = {
    1:  "Address → Coordinates",
    2:  "Coordinates → Address",
    3:  "Top News Headlines by Region",
    4:  "IP Address → Geolocation",
    5:  "WHOIS Domain History",
    6:  "Email Validation",
    7:  "Sentiment Analysis",
    8:  "Language Identification",
    9:  "QR Code Generation",
    10: "Multi-Day Weather Forecast"
}
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
    # API‐bias rates
    for cid, pd in counts_api[m].items():
        total = sum(pd.values())
        rates_api[m][cid] = {i: pd[i]/total for i in pd} if total else {}
    # Position‐bias rates
    for cid, pd in counts_pos[m].items():
        total = sum(pd.values())
        rates_pos[m][cid] = {i: pd[i]/total for i in pd} if total else {}

with open('rates_api.json', 'w', encoding='utf-8') as f:
    json.dump(rates_api, f, indent=2, ensure_ascii=False)

# Compute TV‐distances & combined metric
tv_api_by_model   = {m: [] for m in STATS_PATHS}
tv_pos_by_model   = {m: [] for m in STATS_PATHS}
tv_combined_by_model = {m: [] for m in STATS_PATHS}

print("Model     Cluster │   D_api   D_pos   D_combined")
print("─" *  50)
for cid, cluster in enumerate(clusters, start=1):
    K = len(cluster)
    uniform = 1.0 / K
    for m in STATS_PATHS:
        p_api = rates_api[m].get(cid, {})
        p_pos = rates_pos[m].get(cid, {})
        # TV distance for API‐bias
        tv_api = 0.5 * sum(abs(p_api.get(i, 0.0) - uniform) for i in range(1, K+1))
        # TV distance for position‐bias
        tv_pos = 0.5 * sum(abs(p_pos.get(i, 0.0) - uniform) for i in range(1, K+1))
        # Combined
        tv_combined = (tv_api + tv_pos) / 2

        tv_api_by_model[m].append(tv_api)
        tv_pos_by_model[m].append(tv_pos)
        tv_combined_by_model[m].append(tv_combined)

        print(f"{m:8s} {cid:7d} │ {tv_api:7.3f} {tv_pos:7.3f} {tv_combined:11.3f}")
    print()

# Finally, average across clusters
print("Average over all clusters:")
for m in STATS_PATHS:
    avg_api   = sum(tv_api_by_model[m])   / len(tv_api_by_model[m])
    avg_pos   = sum(tv_pos_by_model[m])   / len(tv_pos_by_model[m])
    avg_comb  = sum(tv_combined_by_model[m]) / len(tv_combined_by_model[m])
    print(f"{m:8s}:  D_api={avg_api:5.3f},  D_pos={avg_pos:5.3f},  D_combined={avg_comb:5.3f}")

models = list(STATS_PATHS.keys())
cluster_ids = [1, 3, 5, 8]
ncols = 4
nrows = int(np.ceil(len(cluster_ids) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), squeeze=False)

bar_width = 0.35
x = np.arange(len(models))

for idx, cid in enumerate(cluster_ids):
    row, col = divmod(idx, ncols)
    ax = axes[row][col]
    api_vals = [tv_api_by_model[m][cid-1] for m in models]
    pos_vals = [tv_pos_by_model[m][cid-1] for m in models]

    ax.bar(x - bar_width/2, api_vals, width=bar_width, label='API bias')
    ax.bar(x + bar_width/2, pos_vals, width=bar_width, label='Positional bias')

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    # ax.set_ylabel(r'$\delta$')
    # ax.set_title(f'{CLUSTER_NAMES.get(cid,"")}', fontsize=10, fontweight='bold')
    ax.set_title(r'\textbf{' + CLUSTER_NAMES[cid] + '}', fontsize=11)
    ax.set_ylim(0, 1.0)

# turn off any unused subplots
for j in range(len(cluster_ids), nrows*ncols):
    r, c = divmod(j, ncols)
    axes[r][c].axis('off')

plt.tight_layout(rect=[0,0.05,1,1])
fig.legend(
    ['$\delta_{\mathrm{API}}$','$\delta_{\mathrm{pos}}$'],
    loc='lower center',
    ncol=2,
    frameon=False,
    fontsize=10
)
plt.savefig('bias_by_model_and_cluster.pdf')
plt.show()