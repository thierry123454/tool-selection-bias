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
    "Base": ["api_selection_stats_gemini.json"],
    "Random": ["api_selection_stats_gemini-rand-id.json", "api_selection_stats_gemini-rand-id-2.json"],
    "Shuffled": ["api_selection_stats_gemini-shuffle-name.json", "api_selection_stats_gemini-shuffle-name-2.json"],
    "Rand. Targ.": ["api_selection_stats_gemini-rand-id-prom.json", "api_selection_stats_gemini-rand-id-prom-2.json"],
    "Desc. + Param.": ["api_selection_stats_gemini-desc-param-scramble.json", "api_selection_stats_gemini-desc-param-scramble-2.json"],
    "Desc.": ["api_selection_stats_gemini-desc-scramble.json"],
    "Param.": ["api_selection_stats_gemini-param-scramble.json"],
}
CLUSTERS_JSON  = "../2_generate_clusters_and_refine/duplicate_api_clusters.json"
BASE = "api_selection_distributions_sample"
OUTPUT_PDF     = BASE + ".pdf"
OUTPUT_PNG     = BASE + ".png"
TITLE = "Distribution of Selected API using Gemini with Random, Shuffled, or Targeted Tool Names."
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

def compute_rates_from_stats(stats):
    """Given one stats list, return position-level selection rates per cluster."""
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

# Plotting
n_clusters = len(clusters)
ncols = 5
nrows = math.ceil(n_clusters / ncols)
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

    for i, name in enumerate(models):
        means = [mean_rates[name].get(idx, {}).get(pos, 0) for pos in x]
        errs  = [std_rates[name].get(idx, {}).get(pos, 0) for pos in x]
        positions = x + bar_w*(i-(n_models-1)/2)
        ax.bar(positions, means, width=bar_w, label=name, yerr=errs if any(e > 0 for e in errs) else None, capsize=3)
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