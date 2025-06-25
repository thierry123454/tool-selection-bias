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
    "ChatGPT": "api_selection_stats_chatgpt.json",
    "Claude":  "api_selection_stats_claude.json",
    "Gemini":  "api_selection_stats_gemini.json",
    "DeepSeek":  "api_selection_stats_deepseek.json",
}
CLUSTERS_JSON  = "2_generate_clusters_and_refine/duplicate_api_clusters.json"
OUTPUT_PDF     = "api_selection_distributions_by_model.pdf"
OUTPUT_PNG     = "api_selection_distributions_by_model.png"
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

# We know there are 10 clusters, so make 2 rows × 5 columns
n_clusters = len(clusters)
ncols = 5
nrows = 2

fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 4*nrows), squeeze=False)
fig.suptitle("Distribution of Selected API Within Each Cluster", fontsize=16)

models = list(STATS_PATHS.keys())
n_models = len(models)
bar_w = 0.8 / n_models

for idx, cluster in enumerate(clusters, start=1):
    row, col = divmod(idx-1, ncols)
    ax = axes[row][col]
    cluster_size = len(cluster)
    x = np.arange(1, cluster_size+1)

    # draw one bar per model, offset horizontally
    for i, name in enumerate(models):
        model_rates = [rates[name].get(idx, {}).get(pos, 0) for pos in x]
        ax.bar(x + bar_w*(i-(n_models-1)/2), model_rates,
               width=bar_w, label=name)

    # x-axis labels
    ax.set_xticks(x)
    tools = [escape_tex(ep["tool"]) for ep in cluster]
    ax.set_xticklabels(tools, rotation=90, ha="right", fontsize=6)
    ax.set_ylim(0, 1.0)
    ax.set_title(CLUSTER_NAMES.get(idx, ""), fontsize=10)

# turn off any unused axes
for ax_row in axes:
    for ax in ax_row:
        if not ax.has_data():
            ax.axis('off')

# shared legend at bottom
fig.legend(models, loc="lower center", ncol=n_models, frameon=False, fontsize=12)
plt.tight_layout(rect=[0,0.05,1,0.95])

# save & show
fig.savefig(OUTPUT_PDF, format="pdf", transparent=True)
print(f"Saved chart grid to {OUTPUT_PDF}")

fig.savefig(OUTPUT_PNG, format="png", transparent=True)
print(f"Saved chart grid to {OUTPUT_PNG}")
plt.show()