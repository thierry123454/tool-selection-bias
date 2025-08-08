import json
import math
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

# Setup LaTeX
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# ─── CONFIG ────────────────────────────────────────────────────────────
STATS_PATHS = {
    "ChatGPT 4.1":  "api_selection_stats_chatgpt_4.json",
    # "Claude":  "api_selection_stats_claude.json",
    "Gemini":  "api_selection_stats_gemini.json",
    "DeepSeek":  "api_selection_stats_deepseek.json",
    # "ToolLLama":  "api_selection_stats_toolllama.json"
    "Qwen":  "api_selection_stats_qwen-235b.json",
}
CLUSTERS_JSON  = "../2_generate_clusters_and_refine/duplicate_api_clusters.json"
OUTPUT_PDF     = "api_selection_distributions_by_model_full.pdf"
OUTPUT_PNG     = "api_selection_distributions_by_model_full.png"

SELECT_CLUSTERS = None # [1, 3, 5, 8]
# ────────────────────────────────────────────────────────────────────────

# LaTeX special chars:  # $ % & ~ _ ^ \ { }
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
            s = s[:15 - 1] + "…"   # chop + ellipsis
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

# Load data
clusters = load_json(CLUSTERS_JSON)
model_stats = {name: load_json(path) for name, path in STATS_PATHS.items()}

# stats entries are [ query_id, cluster_id, pos_in_cluster, pos_in_relevant_list ]
# build per-cluster lists
counts = {}
for model, stats in model_stats.items():
    counts[model] = {}
    for _, cid, pos, _ in stats:
        counts[model].setdefault(cid, {})
        counts[model][cid].setdefault(pos, 0)
        counts[model][cid][pos] += 1

rates = {}
for name, clusterdict in counts.items():
    rates[name] = {}
    for cid, posdict in clusterdict.items():
        total = sum(posdict.values())
        if total > 0:
            rates[name][cid] = {pos: cnt/total for pos, cnt in posdict.items()}
        else:
            rates[name][cid] = {}

# Determine which clusters to plot
if SELECT_CLUSTERS is None:
    cluster_ids = list(range(1, len(clusters) + 1))
else:
    cluster_ids = sorted([cid for cid in SELECT_CLUSTERS if 1 <= cid <= len(clusters)])
if not cluster_ids:
    raise ValueError("No valid cluster IDs to plot.")

clusters_to_plot = [clusters[cid - 1] for cid in cluster_ids]

# Layout: up to 5 columns to match previous style
n_clusters = len(clusters_to_plot)
ncols = min(5, n_clusters)
nrows = math.ceil(n_clusters / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols * 1.25, 4 * nrows * 1.25), squeeze=False)

models = list(STATS_PATHS.keys())
n_models = len(models)
bar_w = 0.8 / n_models
legend_handles = None

for plot_idx, (cid, cluster) in enumerate(zip(cluster_ids, clusters_to_plot), start=1):
    row, col = divmod(plot_idx - 1, ncols)
    ax = axes[row][col]
    cluster_size = len(cluster)
    x = np.arange(1, cluster_size + 1)

    # Draw bars and on the first subplot collect the bar containers for legend
    containers = []
    for i, name in enumerate(models):
        rates_for_model = [rates[name].get(cid, {}).get(pos, 0) for pos in x]
        cont = ax.bar(
            x + bar_w*(i - (len(models)-1)/2),
            rates_for_model,
            width=bar_w,
            label=name
        )
        if plot_idx == 1:
            containers.append(cont)

    # add horizontal lines at 0.2,0.4,0.6,0.8
    for y in [0.4, 0.6, 0.8]:
        ax.axhline(y=y, color='gray', linestyle='--', linewidth=0.5)
    # extra-thick line at y=0.2
    ax.axhline(y=0.2, color='black', linestyle='--', linewidth=1)

    ax.set_xticks(x)
    tools = [escape_tex(ep["tool"]) for ep in cluster]
    ax.set_xticklabels(tools, rotation=45, ha="right", fontsize=14)
    ax.set_ylim(0, 1.0)
    # ax.set_title(CLUSTER_NAMES.get(cid, f"Cluster {cid}"), fontsize=13, fontweight='bold')
    ax.set_title(r'\textbf{' + CLUSTER_NAMES[cid] + '}', fontsize=17)
        # increase tick label sizes for readability
    ax.tick_params(axis='y', labelsize=11)
    if plot_idx == 1:
        ax.set_ylabel("Selection Rate", fontsize=20)
        ax.tick_params(axis='y', labelsize=20)
        # Save containers for legend
        legend_handles = containers
    else:
        ax.set_yticks([])

# turn off any unused axes
for ax_row in axes:
    for ax in ax_row:
        if not ax.has_data():
            ax.axis('off')

# shared legend at bottom
fig.legend(
    legend_handles,
    models,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.02),
    ncol=len(models),
    frameon=False,
    fontsize=14
)

# fig.subplots_adjust(bottom=0.2, top=0.88, hspace=0.4, wspace=0.3)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])

# save & show
fig.savefig(OUTPUT_PDF, format="pdf", transparent=True)
print(f"Saved chart grid to {OUTPUT_PDF}")

fig.savefig(OUTPUT_PNG, format="png", transparent=True)
print(f"Saved chart grid to {OUTPUT_PNG}")
plt.show()