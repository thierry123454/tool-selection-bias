import json
import os
import math
import matplotlib.pyplot as plt
from collections import defaultdict

# Setup LaTeX
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# ─── CONFIG ────────────────────────────────────────────────────────────
SELECTION_PATH = "api_selection_stats_claude.json"
CLUSTERS_JSON   = "../2_generate_clusters_and_refine/duplicate_api_clusters.json"
# ────────────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# Load data
stats = load_json(SELECTION_PATH)

# stats entries are [ query_id, cluster_id, pos_in_cluster, pos_in_relevant_list ]
# build per-cluster lists
overall_query_pos = []

for _, _, _, pos_query in stats:
    overall_query_pos.append(pos_query)

# Plot overall histogram of position in relevant-APIs list
plt.figure(figsize=(6,4))
max_qpos = max(overall_query_pos) if overall_query_pos else 1
bins = range(1, max_qpos+2)
plt.hist(overall_query_pos, bins=bins, align='left', rwidth=0.8)
plt.xticks(range(1, max_qpos+1))
plt.xlabel("Position")
plt.ylabel("Count")
plt.title("Overall Distribution of Selected API")
plt.tight_layout()

pdf_path = "selected_position.pdf"
plt.savefig(pdf_path, format="pdf")
print(f"Saved histogram grid to {pdf_path}")


plt.show()