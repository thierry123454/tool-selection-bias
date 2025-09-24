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
    "ChatGPT 4.1":  ["selection_stats/api_selection_stats_chatgpt_4.json", "selection_stats/api_selection_stats_chatgpt_4-2.json", "selection_stats/api_selection_stats_chatgpt_4-3.json"],
    "Claude":       ["selection_stats/api_selection_stats_claude.json", "selection_stats/api_selection_stats_claude-2.json", "selection_stats/api_selection_stats_claude-3.json"],
    "Gemini":       ["selection_stats/api_selection_stats_gemini.json", "selection_stats/api_selection_stats_gemini-2.json", "selection_stats/api_selection_stats_gemini-3.json"],
    "DeepSeek":     ["selection_stats/api_selection_stats_deepseek.json", "selection_stats/api_selection_stats_deepseek-2.json", "selection_stats/api_selection_stats_deepseek-3.json"],
    "ToolLLaMA":    ["selection_stats/api_selection_stats_toolllama.json", "selection_stats/api_selection_stats_toolllama-2.json", "selection_stats/api_selection_stats_toolllama-3.json"],
    "Qwen3 (235B)":         ["selection_stats/api_selection_stats_qwen-235b.json", "selection_stats/api_selection_stats_qwen-235b-2.json", "selection_stats/api_selection_stats_qwen-235b-3.json"]
}
CLUSTERS_JSON  = "../2_generate_clusters_and_refine/duplicate_api_clusters.json"
OUTPUT_PDF     = "api_selection_distributions_by_model_subset.pdf"
OUTPUT_PNG     = "api_selection_distributions_by_model_subset.png"
SELECT_CLUSTERS = [1, 3, 8]
MODEL_COLORS = {
   "Gemini":     "#4C78A8",  # blue
   "ChatGPT 3.5":    "#BC6713",  # darker orange
   "ChatGPT 4.1":    "#F58518",  # orange
   "Claude":     "#B279A2",  # purple
   "DeepSeek":   "#E45756",  # red
   "Qwen3 (235B)":       "#72B7B2",  # teal
   "ToolLLaMA":  "#9D755D",  # brown
}
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

# Load data
clusters = load_json(CLUSTERS_JSON)

# Load stats: list-of-runs per model
model_runs = {}
for model_name, paths in STATS_PATHS.items():
    if isinstance(paths, str):
        paths = [paths]
    model_runs[model_name] = [load_json(p) for p in paths]

# stats entries are [ query_id, cluster_id, pos_in_cluster, pos_in_relevant_list ]
# build per-cluster lists
counts_runs = {}
for model_name, runs in model_runs.items():
    counts_runs[model_name] = []
    for stats in runs:
        cdict = defaultdict(lambda: defaultdict(int))   # {cid: {pos: count}}
        for _, cid, pos, _ in stats:
            cdict[cid][pos] += 1
        counts_runs[model_name].append(cdict)

rates_runs = {}
for model_name, run_counts in counts_runs.items():
    rates_runs[model_name] = []
    for cdict in run_counts:
        rdict = {}
        for cid, posdict in cdict.items():
            total = sum(posdict.values())
            if total > 0:
                rdict[cid] = {pos: cnt / total for pos, cnt in posdict.items()}
            else:
                rdict[cid] = {}
        rates_runs[model_name].append(rdict)

means = {}
stds = {}
for model_name, run_rates in rates_runs.items():
    means[model_name] = {}
    stds[model_name] = {}
    n_runs = len(run_rates)
    # enumerate all cluster ids present in any run for this model
    cids = set()
    for rr in run_rates:
        cids.update(rr.keys())
    for cid in cids:
        means[model_name][cid] = {}
        stds[model_name][cid] = {}
        # figure out max position for this cluster from clusters JSON
        cluster_size = len(clusters[cid - 1])
        for pos in range(1, cluster_size + 1):
            vals = []
            for rr in run_rates:
                vals.append(rr.get(cid, {}).get(pos, 0.0))
            vals = np.array(vals, dtype=float)
            means[model_name][cid][pos] = float(np.mean(vals))
            if n_runs > 1:
                stds[model_name][cid][pos] = float(np.std(vals, ddof=1))
            else:
                stds[model_name][cid][pos] = 0.0

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
    # 3 rows: first 8 in a 4x2 grid, last 2 each span two columns
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
                                 figsize=(3.8 * ncols, 5.5 * nrows),
                                 squeeze=False)
    axes = [ax for row in sub_axes for ax in row][:len(clusters_to_plot)]

for plot_idx, (cid, cluster, ax) in enumerate(zip(cluster_ids, clusters_to_plot, axes), start=1):
    cluster_size = len(cluster)
    x = np.arange(1, cluster_size + 1)

    containers = []
    for i, name in enumerate(models):
        mean_vals = [means[name].get(cid, {}).get(pos, 0.0) for pos in x]
        std_vals  = [stds[name].get(cid, {}).get(pos, 0.0) for pos in x]

        cont = ax.bar(
            x + bar_w * (i - (len(models) - 1) / 2),
            mean_vals,
            yerr=std_vals,
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

    ax.set_xticklabels(tools, rotation=(90 if SELECT_CLUSTERS else 30), ha="right", fontsize=(12 if SELECT_CLUSTERS else 9))
    ax.set_ylim(0, 1.0)
    ax.set_title(r'\textbf{' + CLUSTER_NAMES[cid] + '}', fontsize=14)

    # only once, from the first subplot
    if legend_handles is None:
        legend_handles, legend_labels = ax.get_legend_handles_labels()

    if plot_idx == 1 or (SELECT_CLUSTERS == None and (plot_idx == 5 or plot_idx == 9)):
        ax.set_ylabel("Selection Rate", fontsize=18)
        ax.tick_params(axis='y', labelsize=15)
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
    bbox_to_anchor=(0.5, 1),
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