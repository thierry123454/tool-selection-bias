import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Setup LaTeX
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

STATS_PATHS = {
    "ChatGPT 3.5": ["api_selection_stats_chatgpt_no_func.json",
                    "api_selection_stats_chatgpt_no_func-2.json",
                    "api_selection_stats_chatgpt_no_func-3.json"],
    "ChatGPT 4.1":  ["api_selection_stats_chatgpt_4.json",
                     "api_selection_stats_chatgpt_4-2.json",
                     "api_selection_stats_chatgpt_4-3.json"],
    "Claude":       ["api_selection_stats_claude.json"],
    "Gemini":       ["api_selection_stats_gemini.json",
                     "api_selection_stats_gemini-2.json",
                     "api_selection_stats_gemini-3.json"],
    "DeepSeek":     ["api_selection_stats_deepseek.json",
                     "api_selection_stats_deepseek-2.json",
                     "api_selection_stats_deepseek-3.json"],
    "ToolLLaMA":    ["api_selection_stats_toolllama.json",
                     "api_selection_stats_toolllama-2.json"],
    "Qwen":         ["api_selection_stats_qwen-235b.json",
                     "api_selection_stats_qwen-235b-2.json",
                     "api_selection_stats_qwen-235b-3.json"]
}
CLUSTERS_JSON = "../2_generate_clusters_and_refine/duplicate_api_clusters.json"

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

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

clusters = load_json(CLUSTERS_JSON)
num_clusters = len(clusters)

# For each model and cluster, store list of TV values across runs
tv_api_by_model = {m: defaultdict(list) for m in STATS_PATHS}
tv_pos_by_model = {m: defaultdict(list) for m in STATS_PATHS}
tv_comb_by_model = {m: defaultdict(list) for m in STATS_PATHS}

# Also store per-run overall means (averaged over clusters) to summarize with mean±std
overall_by_model = {m: {"api": [], "pos": [], "comb": []} for m in STATS_PATHS}

for model, run_paths in STATS_PATHS.items():
    for run_path in run_paths:
        stats = load_json(run_path)

        # Build counts for this single run
        counts_api = defaultdict(lambda: defaultdict(int))  # cid -> pos_in_cluster -> count
        counts_pos = defaultdict(lambda: defaultdict(int))  # cid -> pos_in_list   -> count

        for _, cid, pos_cluster, pos_list in stats:
            counts_api[cid][pos_cluster] += 1
            counts_pos[cid][pos_list]    += 1

        # Compute TV per cluster for this run
        run_tv_api_vals, run_tv_pos_vals, run_tv_comb_vals = [], [], []

        for cid, cluster in enumerate(clusters, start=1):
            K = len(cluster)
            uniform = 1.0 / K

            # Normalize to rates for this cluster in this run
            total_api = sum(counts_api[cid].values())
            total_pos = sum(counts_pos[cid].values())

            p_api = {i: (counts_api[cid][i] / total_api) if total_api else 0.0 for i in range(1, K+1)}
            p_pos = {i: (counts_pos[cid][i] / total_pos) if total_pos else 0.0 for i in range(1, K+1)}

            tv_api = 0.5 * sum(abs(p_api[i] - uniform) for i in range(1, K+1))
            tv_pos = 0.5 * sum(abs(p_pos[i] - uniform) for i in range(1, K+1))
            tv_comb = (tv_api + tv_pos) / 2.0

            tv_api_by_model[model][cid].append(tv_api)
            tv_pos_by_model[model][cid].append(tv_pos)
            tv_comb_by_model[model][cid].append(tv_comb)

            run_tv_api_vals.append(tv_api)
            run_tv_pos_vals.append(tv_pos)
            run_tv_comb_vals.append(tv_comb)

        # Per-run overall averages (averaged over clusters)
        overall_by_model[model]["api"].append(float(np.mean(run_tv_api_vals)))
        overall_by_model[model]["pos"].append(float(np.mean(run_tv_pos_vals)))
        overall_by_model[model]["comb"].append(float(np.mean(run_tv_comb_vals)))

# ── Printing: per-cluster mean±std across runs ─────────────────────────
print("Per-cluster TV distance (mean ± std) across runs")
print("Model           Cluster │      D_api           D_pos         D_combined")
print("─" * 75)
for cid in range(1, num_clusters + 1):
    for m in STATS_PATHS:
        api_vals = tv_api_by_model[m][cid]
        pos_vals = tv_pos_by_model[m][cid]
        comb_vals = tv_comb_by_model[m][cid]

        api_mean, api_std   = np.mean(api_vals), np.std(api_vals)   if api_vals else (np.nan, np.nan)
        pos_mean, pos_std   = np.mean(pos_vals), np.std(pos_vals)   if pos_vals else (np.nan, np.nan)
        comb_mean, comb_std = np.mean(comb_vals), np.std(comb_vals) if comb_vals else (np.nan, np.nan)

        print(f"{m:15s} {cid:7d} │  {api_mean:6.3f} ± {api_std:5.3f}   "
              f"{pos_mean:6.3f} ± {pos_std:5.3f}   {comb_mean:6.3f} ± {comb_std:5.3f}")
    print()

# ── Printing: overall (per-run means averaged) mean±std across runs ────
print("Overall (average across clusters) — mean ± std across runs")
for m in STATS_PATHS:
    a = np.array(overall_by_model[m]["api"])
    p = np.array(overall_by_model[m]["pos"])
    c = np.array(overall_by_model[m]["comb"])

    print(f"{m:15s}:  D_api={a.mean():.3f} ± {a.std():.3f},  "
          f"D_pos={p.mean():.3f} ± {p.std():.3f},  D_combined={c.mean():.3f} ± {c.std():.3f}")

models = list(STATS_PATHS.keys())
cluster_ids = [2, 6, 9]
ncols = len(cluster_ids)
nrows = int(np.ceil(len(cluster_ids) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows), squeeze=False)

bar_width = 0.35
x = np.arange(len(models))

for idx, cid in enumerate(cluster_ids):
    row, col = divmod(idx, ncols)
    ax = axes[row][col]
    api_vals = [tv_api_by_model[m][cid][0] for m in models]
    pos_vals = [tv_pos_by_model[m][cid][0] for m in models]

    ax.bar(x - bar_width/2, api_vals, width=bar_width, label='API bias')
    ax.bar(x + bar_width/2, pos_vals, width=bar_width, label='Positional bias')

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=12)
    # ax.set_ylabel(r'$\delta$')
    # ax.set_title(f'{CLUSTER_NAMES.get(cid,"")}', fontsize=10, fontweight='bold')
    ax.set_title(r'\textbf{' + CLUSTER_NAMES[cid] + '}', fontsize=15)
    ax.set_ylim(0, 1.0)

    if idx == 0:
        ax.set_ylabel("TV distance vs. Uniform", fontsize=15)
        ax.tick_params(axis='y', labelsize=14)
    else:
        ax.set_yticks([])

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
    fontsize=14
)
plt.savefig('bias_by_model_and_cluster.pdf')
plt.savefig('bias_by_model_and_cluster.png')
plt.show()