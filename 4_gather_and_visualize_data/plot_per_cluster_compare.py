import json
import math
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

# Setup LaTeX
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# ─── CONFIG ────────────────────────────────────────────────────────────

# SIZE
# STATS_PATHS = {
#     "1.7B": "api_selection_stats_qwen-1.7b.json",
#     "4B":  "api_selection_stats_qwen-4b.json",
#     "8B":  "api_selection_stats_qwen-8b.json"
# }
# CLUSTERS_JSON  = "../2_generate_clusters_and_refine/duplicate_api_clusters.json"
# BASE = "api_selection_distributions_size"
# OUTPUT_PDF     = BASE + ".pdf"
# OUTPUT_PNG     = BASE + ".png"
# TITLE = "Distribution of Selected API using Qwen with Different Model Sizes."

# TEMPERATURE
# STATS_PATHS = {
#     "0": "api_selection_stats_chatgpt-temp-0.json",
#     "0.5":  "api_selection_stats_chatgpt_base.json",
#     "1":  "api_selection_stats_chatgpt-temp-1.json",
#     "2":  "api_selection_stats_chatgpt-temp-2.json"
# }
# CLUSTERS_JSON  = "../2_generate_clusters_and_refine/duplicate_api_clusters.json"
# BASE = "api_selection_distributions_temp"
# OUTPUT_PDF     = BASE + ".pdf"
# OUTPUT_PNG     = BASE + ".png"
# TITLE = "Distribution of Selected API using ChatGPT with Different Temperatures."

# TOP-P
# STATS_PATHS = {
#     "0.7": "api_selection_stats_chatgpt-top-p-0.7.json",
#     "0.9": "api_selection_stats_chatgpt-top-p-0.9.json",
#     "1":  "api_selection_stats_chatgpt_base.json"
# }
# CLUSTERS_JSON  = "../2_generate_clusters_and_refine/duplicate_api_clusters.json"
# BASE = "api_selection_distributions_top_p"
# OUTPUT_PDF     = BASE + ".pdf"
# OUTPUT_PNG     = BASE + ".png"
# TITLE = "Distribution of Selected API using ChatGPT with Different Top-$p$."

# SHUFFLE EXPERIMENT
STATS_PATHS = {
    "Base": "api_selection_stats_gemini.json",
    "Random": "api_selection_stats_gemini-rand-id.json",
    "Shuffled": "api_selection_stats_gemini-shuffle-name.json",
    "Rand. Targ.": "api_selection_stats_gemini-rand-id-prom.json",
}
CLUSTERS_JSON  = "../2_generate_clusters_and_refine/duplicate_api_clusters.json"
BASE = "api_selection_distributions_sample"
OUTPUT_PDF     = BASE + ".pdf"
OUTPUT_PNG     = BASE + ".png"
TITLE = "Distribution of Selected API using Gemini with Random, Shuffled, or Targeted Tool Names."

# ZERO TEMPERATURE
# STATS_PATHS = {
#     "1": "api_selection_stats_chatgpt-temp-0.json",
#     "2": "api_selection_stats_chatgpt-temp-0-1.json",
#     "3": "api_selection_stats_chatgpt-temp-0-2.json"
# }
# CLUSTERS_JSON  = "../2_generate_clusters_and_refine/duplicate_api_clusters.json"
# BASE = "api_selection_distributions_temp_0"
# OUTPUT_PDF     = BASE + ".pdf"
# OUTPUT_PNG     = BASE + ".png"
# TITLE = "Distribution of Selected API using ChatGPT with Temperature 0."

# SHUFFLE VS. CYCLIC
# STATS_PATHS = {
#     "Random": "api_selection_stats_chatgpt_random.json",
#     "Cyclic":  "api_selection_stats_chatgpt_base.json"
# }
# CLUSTERS_JSON  = "../2_generate_clusters_and_refine/duplicate_api_clusters.json"
# BASE = "api_selection_distributions_cyclic_vs_random"
# OUTPUT_PDF     = BASE + ".pdf"
# OUTPUT_PNG     = BASE + ".png"
# TITLE = "Distribution of Selected API using ChatGPT with Different Temperatures."

# SYSTEM PROMPTS
# STATS_PATHS = {
#     "Base": "api_selection_stats_chatgpt_base.json",
#     "Similar":  "api_selection_stats_chatgpt_sim.json",
#     "Adjusted":  "api_selection_stats_chatgpt_adj.json"
# }
# CLUSTERS_JSON  = "../2_generate_clusters_and_refine/duplicate_api_clusters.json"
# BASE = "api_selection_distributions_prompts"
# OUTPUT_PDF     = BASE + ".pdf"
# OUTPUT_PNG     = BASE + ".png"

# TITLE = "Distribution of Selected API Position using ChatGPT with different System Prompts."
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
# fig.suptitle(TITLE, fontsize=16)

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


print("\n=== Selection distributions (text) ===")
for idx, cluster in enumerate(clusters, start=1):
    cluster_name = CLUSTER_NAMES.get(idx, f"Cluster {idx}")
    print(f"\nCluster {idx}: {cluster_name}")
    tools = [ep["tool"] for ep in cluster]
    for model in models:
        print(f"  Model: {model}")
        pos_rates = rates[model].get(idx, {})
        for pos in range(1, len(cluster) + 1):
            rate = pos_rates.get(pos, 0.0)
            tool = tools[pos - 1]
            print(f"    {pos:2d}. {tool:30s} : {rate:.3f}")
