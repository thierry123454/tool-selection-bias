#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Setup LaTeX
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# ─── CONFIG ────────────────────────────────────────────────────────────
STATS_PATH_A    = "api_selection_stats_gemini-sample.json"
STATS_PATH_B    = "api_selection_stats_gemini-sample-temp-2.json"
CLUSTERS_JSON   = "../2_generate_clusters_and_refine/duplicate_api_clusters.json"
OUTPUT_PDF      = "query_api_selection_rates.pdf"
OUTPUT_PNG      = "query_api_selection_rates.png"
FIG_TITLE       = "Gemini: API Selection over 20 Runs per Query"
QUERY_TITLES = {
    1: "What are the latitude and longitude for 1600 Amphitheatre Parkway, Mountain View, CA?",
    2: "Get the latitude and longitude for 20 W 34th St, New York, NY.",
    3: "Detect the language of 'Salve, quid agis hodie?'",
    4: "What language is used in '{JAPANESE SENTENCE}'?"
}
# ────────────────────────────────────────────────────────────────────────

TEX_ESCAPES = {
    '&':  r'\&', '%':  r'\%', '$':  r'\$', '#':  r'\#',
    '_':  r'\_', '{':  r'\{', '}':  r'\}', '~':  r'\textasciitilde{}',
    '^':  r'\^{}', '\\': r'\textbackslash{}',
}

def escape_tex(s):
    if len(s) > 15:
        s = s[:14] + "…"   # chop + ellipsis
    return ''.join(TEX_ESCAPES.get(ch, ch) for ch in s)
def load_and_block(path):
    """Load JSON stats and split into 4 blocks of 20 runs each."""
    with open(path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    blocks = {i: [] for i in range(1,5)}
    for query_id, cid, selected_api, _ in stats:
        block = (query_id - 1) // 20 + 1
        if 1 <= block <= 4:
            blocks[block].append((cid, selected_api))
    return blocks

# Load clusters & both stats‐blocks
with open(CLUSTERS_JSON, "r", encoding="utf-8") as f:
    clusters = json.load(f)

blocks_A = load_and_block(STATS_PATH_A)
blocks_B = load_and_block(STATS_PATH_B)

# 3) Plot side‐by‐side
fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
fig.suptitle(FIG_TITLE, fontsize=18)

for block_idx in range(1, 5):
    ax = axes[(block_idx-1)//2][(block_idx-1)%2]
    a_entries = blocks_A[block_idx]
    b_entries = blocks_B[block_idx]
    if not a_entries:
        ax.axis('off')
        continue

    # they share the same cluster
    cid = a_entries[0][0]
    tools = clusters[cid-1]
    k     = len(tools)
    x     = np.arange(1, k+1)

    # compute rates for each dataset
    def rates_from(entries):
        picks = [sel for _, sel in entries]
        cnts  = Counter(picks)
        total = len(picks)
        return [cnts.get(i,0)/total for i in x]

    ratesA = rates_from(a_entries)
    ratesB = rates_from(b_entries)

    # bar width and offsets
    w = 0.35
    ax.bar(x - w/2, ratesA, width=w, label="Temp: 0.5", color="C0")
    ax.bar(x + w/2, ratesB, width=w, label="Temp: 2",   color="C1")

    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    labels = [escape_tex(t["tool"]) for t in tools]
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(QUERY_TITLES[block_idx], fontsize=10)

    if block_idx == 1:
        ax.legend(loc="upper right")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig(OUTPUT_PDF, dpi=150)
fig.savefig(OUTPUT_PNG, dpi=150)
plt.show()