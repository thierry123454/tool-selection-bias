import json
import math
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from matplotlib.patches import Patch


MODEL_COLORS = {
    "Base": "#7f7f7f",
    "Rand. Name": "#1f77b4",
    "Desc. + Param.": "#ff7f0e",
    "Targ. Desc.": "#2ca02c",
    "Swap. Desc.": "#d62728",
    "AbO": "#000000",
    "Full": "#ff00f2",
}

# Setup LaTeX
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# ─── CONFIG ────────────────────────────────────────────────────────────
STATS_PATHS = {
    "Base": ["selection_stats/api_selection_stats_gemini.json"],
    "Rand. Name": ["selection_stats/api_selection_stats_gemini-rand-id.json", "selection_stats/api_selection_stats_gemini-rand-id-2.json"],
    "Desc. + Param.": ["selection_stats/api_selection_stats_gemini-desc-param-scramble.json", "selection_stats/api_selection_stats_gemini-desc-param-scramble-2.json"],
    "Targ. Desc.": ["selection_stats/api_selection_stats_gemini_desc_prom.json", "selection_stats/api_selection_stats_gemini_desc_prom-2.json"],
    "Swap. Desc.": ["selection_stats/api_selection_stats_answer_gemini_desc_swap.json", "selection_stats/api_selection_stats_answer_gemini_desc_swap-2.json"],
    "AbO": ["selection_stats/api_selection_stats_gemini_abo.json"],
    "Full": ["selection_stats/api_selection_stats_gemini_full_scramble.json"]
}
CLUSTERS_JSON  = "../2_generate_clusters_and_refine/duplicate_api_clusters.json"
BASE = "api_selection_distributions_perturbation"
OUTPUT_PDF     = BASE + ".pdf"
OUTPUT_PNG     = BASE + ".png"
DESC_SWAP_JSON = "../5_bias_investigation/experiments/desc_swap_1.json"
TITLE = "Distribution of Selected API using Gemini with Random, Shuffled, or Targeted Tool Names."
SELECT_CLUSTERS = [1, 4, 5, 6, 8, 10]
# ────────────────────────────────────────────────────────────────────────

# LaTeX special chars
TEX_ESCAPES = {
    '&':  r'\&',
    '%':  r'\%',
    '$':  r'\$',
    '#':  r'\#',
    '_':  r'\_',
    '{':  r'\{',
    '}':  r'\}',
    '~':  r'\textasciitilde{}',
    '^':  r'\^{}',
    '\\': r'\textbackslash{}',
}

def escape_tex(s):
    if len(s) > 15:
        s = s[:15 - 1] + "..."
    return ''.join(TEX_ESCAPES.get(ch, ch) for ch in s)

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

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_rates_from_stats(stats):
    """Given one stats list, return API-level selection rates per cluster."""
    counts = defaultdict(lambda: defaultdict(int))
    for _, cid, pos, _ in stats:
        counts[cid][pos] += 1
    rates = {}
    for cid in counts:
        total = sum(counts[cid].values())
        if total > 0:
            rates[cid] = {i: counts[cid][i]/total for i in counts[cid]}
        else:
            rates[cid] = {}
    return rates

# Load clusters
clusters = load_json(CLUSTERS_JSON)
swap_pairs = load_json(DESC_SWAP_JSON)
most_set   = set(swap_pairs.keys())
least_set  = set(swap_pairs.values())

# Normalize STATS_PATHS so each value is a list
for k, v in list(STATS_PATHS.items()):
    if isinstance(v, str):
        STATS_PATHS[k] = [v]

# Compute per-run rates, then mean+std
model_runs = {}  # model -> list of per-run rates
for model, paths in STATS_PATHS.items():
    per_run = []
    for p in paths:
        stats = load_json(p)
        per_run.append(compute_rates_from_stats(stats))
    model_runs[model] = per_run

print(model_runs)

# Aggregate: mean and std across runs
mean_rates = {}
std_rates = {}
for model, runs in model_runs.items():
    mean_rates[model] = {}
    std_rates[model] = {}
    for cid, cluster in enumerate(clusters, start=1):
        K = len(cluster)
        mean_rates[model].setdefault(cid, {})
        std_rates[model].setdefault(cid, {})
        for pos in range(1, K+1):
            vals = [ r.get(cid, {}).get(pos, 0.0) for r in runs ]
            mean = np.mean(vals)
            std = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
            mean_rates[model][cid][pos] = mean
            std_rates[model][cid][pos] = std

# Determine which clusters to plot
if SELECT_CLUSTERS is None:
    cluster_ids = list(range(1, len(clusters) + 1))
else:
    cluster_ids = sorted([cid for cid in SELECT_CLUSTERS if 1 <= cid <= len(clusters)])
if not cluster_ids:
    raise ValueError("No valid cluster IDs to plot.")

clusters_to_plot = [clusters[cid - 1] for cid in cluster_ids]

n_clusters = len(clusters_to_plot)
ncols = 3
nrows = 2

# wider per subplot
FIG_W_PER_COL = 4.5
FIG_H_PER_ROW = 5.0
fig, axes = plt.subplots(nrows, ncols,
                         figsize=(FIG_W_PER_COL*ncols, FIG_H_PER_ROW*nrows),
                         squeeze=False)

# fig.suptitle(TITLE, fontsize=16)

models = list(STATS_PATHS.keys())
n_models = len(models)
bar_w = 0.8 / n_models

for plot_idx, (cid, cluster) in enumerate(zip(cluster_ids, clusters_to_plot), start=1):
    row, col = divmod(plot_idx-1, ncols)
    ax = axes[row][col]
    cluster_size = len(cluster)
    x = np.arange(1, cluster_size+1)

    for i, name in enumerate(models):
        means = [mean_rates[name].get(cid, {}).get(pos, 0) for pos in x]
        errs  = [std_rates[name].get(cid, {}).get(pos, 0) for pos in x]
        positions = x + bar_w*(i-(n_models-1)/2)
        ax.bar(
            positions, means, width=bar_w,
            yerr=errs if any(e > 0 for e in errs) else None,
            capsize=3,
            color=MODEL_COLORS.get(name, None),
            label=name
        )
    
    # add horizontal lines at 0.2,0.4,0.6,0.8
    for y in [0.4, 0.6, 0.8]:
        ax.axhline(y=y, color='gray', linestyle='--', linewidth=0.5)
    # extra-thick line at y=0.2
    ax.axhline(y=0.2, color='black', linestyle='--', linewidth=1)
    
    # x-axis labels
    ax.set_xticks(x)
    tools_raw = [ep["tool"] for ep in cluster]
    tools = [escape_tex(t) for t in tools_raw]
    ax.set_xticklabels(tools, rotation=45, ha="right", fontsize=14)
    ax.set_ylim(0, 1.0)
    ax.set_title(r'\textbf{' + CLUSTER_NAMES[cid] + '}', fontsize=15)

    for j, tick in enumerate(ax.get_xticklabels()):
        raw_tool = tools_raw[j]
        if raw_tool in most_set:
            tick.set_color('green')
            tick.set_fontweight('bold')
        elif raw_tool in least_set:
            tick.set_color('red')
            tick.set_fontstyle('italic')

    if plot_idx == 1 or plot_idx == 4:
        ax.set_ylabel("Selection Rate", fontsize=14)
        ax.tick_params(axis='y', labelsize=14)
    else:
        ax.set_yticks([])


# turn off any unused axes
for ax_row in axes:
    for ax in ax_row:
        if not ax.has_data():
            ax.axis('off')

# shared legend at top
handles = [Patch(facecolor=MODEL_COLORS[m], label=m) for m in models]
fig.legend(
    handles=handles,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.00),
    ncol=len(models),
    frameon=False,
    fontsize=14
)

plt.tight_layout(rect=[0, 0.05, 1, 0.93])

# save & show
fig.savefig(OUTPUT_PDF, format="pdf", transparent=True)
print(f"Saved chart grid to {OUTPUT_PDF}")

fig.savefig(OUTPUT_PNG, format="png", transparent=True)
print(f"Saved chart grid to {OUTPUT_PNG}")
plt.show()

# Text summary
print("\n=== Selection distributions (text) ===")
for idx, cluster in enumerate(clusters, start=1):
    cluster_name = CLUSTER_NAMES.get(idx, f"Cluster {idx}")
    print(f"\nCluster {idx}: {cluster_name}")
    tools = [ep["tool"] for ep in cluster]
    for model in models:
        print(f"  Model: {model}")
        for pos in range(1, len(cluster) + 1):
            rate_mean = mean_rates[model][idx].get(pos, 0.0)
            rate_std = std_rates[model][idx].get(pos, 0.0)
            tool = tools[pos - 1]
            if rate_std > 0:
                print(f"    {pos:2d}. {tool:30s} : {rate_mean:.3f} ±{rate_std:.3f}")
            else:
                print(f"    {pos:2d}. {tool:30s} : {rate_mean:.3f}")