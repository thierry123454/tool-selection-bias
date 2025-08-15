import json
import math
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from matplotlib.gridspec import GridSpec

# Setup LaTeX
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# ─── CONFIG ────────────────────────────────────────────────────────────
STATS_PATHS = {
    "ChatGPT 4.1":  "api_selection_stats_chatgpt_4.json",
    "Claude":  "api_selection_stats_claude.json",
    "Gemini":  "api_selection_stats_gemini.json",
    "DeepSeek":  "api_selection_stats_deepseek.json",
    "ToolLLaMA":  "api_selection_stats_toolllama.json",
    "Qwen (8B)":  "api_selection_stats_qwen-8b.json"
}
CLUSTERS_JSON  = "../2_generate_clusters_and_refine/duplicate_api_clusters.json"
OUTPUT_PDF     = "api_selection_distributions_by_model_full.pdf"
OUTPUT_PNG     = "api_selection_distributions_by_model_full.png"
SELECT_CLUSTERS = None # [1, 3, 8]
MODEL_COLORS = {
   "Gemini":     "#4C78A8",  # blue
   "ChatGPT 3.5":    "#BC6713",  # darker orange
   "ChatGPT 4.1":    "#F58518",  # orange
   "Claude":     "#B279A2",  # purple
   "DeepSeek":   "#E45756",  # red
   "Qwen":       "#72B7B2",  # teal
   "ToolLLaMA":  "#9D755D",  # brown
}
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
models = list(STATS_PATHS.keys())
n_models = len(models)
bar_w = 0.8 / n_models

axes = []
legend_handles = legend_labels = None

if SELECT_CLUSTERS is None and len(cluster_ids) == 10:
    # 3 rows: first 8 in a 4×2 grid, last 2 each span two columns
    fig = plt.figure(figsize=(16, 11))
    gs = GridSpec(3, 4, figure=fig, wspace=0.28, hspace=0.48)

    axes = []
    # first 8 clusters: rows 0–1, 4 columns
    for idx in range(8):
        r, c = divmod(idx, 4)
        axes.append(fig.add_subplot(gs[r, c]))

    # last 2 clusters: row 2, span cols 0–1 and 2–3 respectively
    axes.append(fig.add_subplot(gs[2, 0:2]))  # spans columns 0 and 1
    axes.append(fig.add_subplot(gs[2, 2:4]))  # spans columns 2 and 3
else:
    # Generic fallback
    n_clusters = len(clusters_to_plot)
    ncols = min(5, n_clusters)
    nrows = math.ceil(n_clusters / ncols)
    fig, sub_axes = plt.subplots(nrows, ncols,
                                 figsize=(3.8 * ncols, 4.2 * nrows),
                                 squeeze=False)
    axes = [ax for row in sub_axes for ax in row][:len(clusters_to_plot)]

for plot_idx, (cid, cluster, ax) in enumerate(zip(cluster_ids, clusters_to_plot, axes), start=1):
    cluster_size = len(cluster)
    x = np.arange(1, cluster_size + 1)

    containers = []
    for i, name in enumerate(models):
        rates_for_model = [rates[name].get(cid, {}).get(pos, 0) for pos in x]
        cont = ax.bar(
            x + bar_w * (i - (len(models) - 1) / 2),
            rates_for_model,
            width=bar_w,
            label=name,
            color=MODEL_COLORS.get(name, None),
        )
        if plot_idx == 1:
            containers.append(cont)

    # light guide lines
    for y in [0.4, 0.6, 0.8]:
        ax.axhline(y=y, color='gray', linestyle='--', linewidth=0.5)
    ax.axhline(y=0.2, color='black', linestyle='--', linewidth=1)

    ax.set_xticks(x)
    tools = [escape_tex(ep["tool"]) for ep in cluster]

    ax.set_xticklabels(tools, rotation=(90 if SELECT_CLUSTERS else 30), ha="right", fontsize=(11 if SELECT_CLUSTERS else 9))
    ax.set_ylim(0, 1.0)
    ax.set_title(r'\textbf{' + CLUSTER_NAMES[cid] + '}', fontsize=14)

    # only once, from the first subplot
    if legend_handles is None:
        legend_handles, legend_labels = ax.get_legend_handles_labels()

    if plot_idx == 1 or (SELECT_CLUSTERS == None and (plot_idx == 5 or plot_idx == 9)):
        ax.set_ylabel("Selection Rate", fontsize=13)
        ax.tick_params(axis='y', labelsize=12)
    else:
        ax.set_yticks([])

# Turn off any unused axes in generic layout
if not (SELECT_CLUSTERS is None and len(cluster_ids) == 10):
    for ax in axes[len(clusters_to_plot):]:
        ax.axis('off')

# Legend
fig.legend(
    legend_handles,
    models,
    loc='upper center',
    bbox_to_anchor=(0.5, 0.98),
    ncol=len(models),
    frameon=False,
    fontsize=13
)
if not SELECT_CLUSTERS:
    fig.subplots_adjust(left=0.0475, right=0.99)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
fig.savefig(OUTPUT_PDF, format="pdf", transparent=True)
fig.savefig(OUTPUT_PNG, format="png", transparent=True)
plt.show()