import json
import math
import matplotlib.pyplot as plt
from collections import defaultdict

# Setup LaTeX
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# ─── CONFIG ────────────────────────────────────────────────────────────
SELECTION_PATH = "api_selection_stats.json"
CLUSTERS_JSON  = "2_generate_clusters_and_refine/duplicate_api_clusters.json"
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
stats    = load_json(SELECTION_PATH)
clusters = load_json(CLUSTERS_JSON)

# stats entries are [ query_id, cluster_id, pos_in_cluster, pos_in_relevant_list ]
# build per-cluster lists
per_cluster = defaultdict(list)
for _, cid, pos_cluster, _ in stats:
    per_cluster[cid].append(pos_cluster)

for cid in per_cluster.keys():
    print(len(per_cluster[cid]))

# We know there are 10 clusters, so make 2 rows × 5 columns
n_clusters = len(per_cluster)
ncols = 5
nrows = 2

fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 4*nrows), squeeze=False)
fig.suptitle("Distribution of Selected API Within Each Cluster", fontsize=16)

for idx, (cid, selections) in enumerate(sorted(per_cluster.items())):
    row, col = divmod(idx, ncols)
    ax = axes[row][col]
    cluster_endpoints = clusters[cid-1]

    cluster_size = len(cluster_endpoints)
    bins = range(1, cluster_size+2)
    ax.hist(selections, bins=bins, align='left', rwidth=0.8, density=True)
    ax.set_xticks(range(1, cluster_size+1))

    tool_names = [ep["tool"] for ep in cluster_endpoints]
    sanitized = [escape_tex(n) for n in tool_names]
    ax.set_xticklabels(sanitized, rotation=45, ha="right", fontsize=6)

    name = CLUSTER_NAMES.get(cid, "")
    ax.set_title(f"{name}")

pdf_path = "api_selection_distributions.pdf"
fig.savefig(pdf_path, format="pdf")
print(f"Saved histogram grid to {pdf_path}")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
for ax_row in axes:
    for ax in ax_row:
        ax.set_ylim(0, 0.6)
plt.show()
