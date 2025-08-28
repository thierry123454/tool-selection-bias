import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.patches import Patch

# LaTeX styling
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# ─── CONFIG ────────────────────────────────────────────────────────────
STATS_PATHS_GEMINI = {
    "Base": ["selection_stats/api_selection_stats_gemini.json"],
    "Rand. Name": ["selection_stats/api_selection_stats_gemini-rand-id.json", "selection_stats/api_selection_stats_gemini-rand-id-2.json", "selection_stats/api_selection_stats_gemini-rand-id-3.json"],
    "Shuff. Name": ["selection_stats/api_selection_stats_gemini-shuffle-name.json", "selection_stats/api_selection_stats_gemini-shuffle-name-2.json", "selection_stats/api_selection_stats_gemini-shuffle-name-3.json"],
    "Targ. Name": ["selection_stats/api_selection_stats_gemini-rand-id-prom.json", "selection_stats/api_selection_stats_gemini-rand-id-prom-2.json", "selection_stats/api_selection_stats_gemini-rand-id-prom-3.json"],
    "Desc. + Param.": ["selection_stats/api_selection_stats_gemini-desc-param-scramble.json", "selection_stats/api_selection_stats_gemini-desc-param-scramble-2.json", "selection_stats/api_selection_stats_gemini-desc-param-scramble-3.json"],
    "Desc.": ["selection_stats/api_selection_stats_gemini-desc-scramble.json", "selection_stats/api_selection_stats_gemini-desc-scramble-2.json", "selection_stats/api_selection_stats_gemini-desc-scramble-3.json"],
    "Targ. Desc.": ["selection_stats/api_selection_stats_gemini_desc_prom.json", "selection_stats/api_selection_stats_gemini_desc_prom-2.json", "selection_stats/api_selection_stats_gemini_desc_prom-3.json"],
    "Param.": ["selection_stats/api_selection_stats_gemini-param-scramble.json", "selection_stats/api_selection_stats_gemini-param-scramble-2.json", "selection_stats/api_selection_stats_gemini-param-scramble-3.json"],
    "Swap. Desc.": ["selection_stats/api_selection_stats_answer_gemini_desc_swap.json", "selection_stats/api_selection_stats_answer_gemini_desc_swap-2.json", "selection_stats/api_selection_stats_gemini_desc_swap-3.json"],
    "Full": ["selection_stats/api_selection_stats_gemini_full_scramble.json"]
}
STATS_PATHS_CHATGPT = {
    "Base": ["selection_stats/api_selection_stats_chatgpt_4.json"],
    "Rand. Name": ["selection_stats/api_selection_stats_chatgpt-rand-id.json", "selection_stats/api_selection_stats_chatgpt-rand-id-2.json"],
    "Desc. + Param.": ["selection_stats/api_selection_stats_chatgpt-desc-param-scramble.json", "selection_stats/api_selection_stats_chatgpt-desc-param-scramble-2.json"]
}
CLUSTERS_JSON  = "../2_generate_clusters_and_refine/duplicate_api_clusters.json"

OUT_PDF = "tv_to_uniform_by_perturbation.pdf"
OUT_PNG = "tv_to_uniform_by_perturbation.png"

K = 5
# ────────────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_rates_from_stats(stats):
    """
    stats entries: [query_id, cluster_id, pos_in_cluster, pos_in_relevant_list]
    returns rates[cid][pos] = selection rate for 'pos' in cluster 'cid'
    """
    counts = defaultdict(lambda: defaultdict(int))
    for _, cid, pos, _ in stats:
        counts[cid][pos] += 1
    rates = {}
    for cid in counts:
        total = sum(counts[cid].values())
        rates[cid] = {i: counts[cid][i] / total for i in counts[cid]} if total else {}
    return rates

def average_dists_across_runs(runs, clusters):
    """
    Average selection distributions across runs for each cluster.
    runs: list of dicts {cid -> {pos -> rate}}
    return: {cid -> {pos -> mean_rate}}
    """
    mean_per_cluster = {}
    for cid, cluster in enumerate(clusters, start=1):
        if not runs:
            continue
        if len(runs[0].get(cid, {})) == 0:
            continue
        pos_means = {}
        for pos in range(1, K + 1):
            vals = [r.get(cid, {}).get(pos, 0.0) for r in runs]
            pos_means[pos] = float(np.mean(vals)) if vals else 0.0
        mean_per_cluster[cid] = pos_means
    return mean_per_cluster

def tv_to_uniform(dist):
    """TV(p, U_K) where U_K is uniform over K items."""
    return 0.5 * sum(abs(dist.get(i, 0.0) - 1.0 / K) for i in range(1, K+1))

def compute_tv_uniform_stats(paths_dict, clusters):
    """
    For each condition (INCLUDING 'Base'):
      - Average across runs to get cond_mean per cluster
      - Compute TV(cond_mean, Uniform_K) for each cluster
      - Return mean/std across clusters
    Output: {cond: (mean_tv, std_tv)}
    """
    paths_dict = {k: (v if isinstance(v, list) else [v]) for k, v in paths_dict.items()}

    cond_runs = {}
    for cond, paths in paths_dict.items():
        runs = []
        for p in paths:
            stats = load_json(p)
            runs.append(compute_rates_from_stats(stats))
        cond_runs[cond] = runs

    out = {}
    for cond, runs in cond_runs.items():
        cond_mean = average_dists_across_runs(runs, clusters)
        tvs = []
        for cid, _ in enumerate(clusters, start=1):
            if cid not in cond_mean:
                continue
            tvs.append(tv_to_uniform(cond_mean[cid]))
        mu = float(np.mean(tvs)) if tvs else 0.0
        sd = float(np.std(tvs, ddof=1)) if len(tvs) > 1 else 0.0
        out[cond] = (mu, sd)
    return out

# Load clusters
clusters = load_json(CLUSTERS_JSON)

# Compute TV-to-uniform stats (means/std across clusters)
gem_stats = compute_tv_uniform_stats(STATS_PATHS_GEMINI, clusters)
chat_stats = compute_tv_uniform_stats(STATS_PATHS_CHATGPT, clusters)

# Order by Gemini (descending), include Base
order = list(STATS_PATHS_GEMINI.keys())
order = sorted(order, key=lambda c: gem_stats.get(c, (0,0))[0], reverse=True)
labels = order

gem_vals = [gem_stats[l][0] if l in gem_stats else 0.0 for l in labels]
gem_errs = [gem_stats[l][1] if l in gem_stats else 0.0 for l in labels]
has_chat = [l in chat_stats for l in labels]
chat_vals = [chat_stats[l][0] if l in chat_stats else None for l in labels]
chat_errs = [chat_stats[l][1] if l in chat_stats else None for l in labels]

# Plot (Gemini + ChatGPT where available)
gem_color  = "#4C78A8"
chat_color = "#F58518"

fig, ax = plt.subplots(figsize=(max(8, 1.6*len(labels)), 5.2))
x = np.arange(len(labels))
w = 0.38

for i, _ in enumerate(labels):
    if has_chat[i]:
        ax.bar(x[i] - w/2, gem_vals[i], width=w, yerr=gem_errs[i], capsize=4, color=gem_color)
        ax.bar(x[i] + w/2, chat_vals[i], width=w, yerr=chat_errs[i], capsize=4, color=chat_color)
    else:
        ax.bar(x[i], gem_vals[i], width=w, yerr=gem_errs[i], capsize=4, color=gem_color)

gem_base_mu = gem_stats.get("Base", (np.nan, 0.0))[0]
if np.isfinite(gem_base_mu):
    ax.axhline(y=gem_base_mu, color=gem_color, linestyle='--', linewidth=1.5, alpha=0.9)

chat_base_mu = chat_stats.get("Base", (np.nan, 0.0))[0] if chat_stats else np.nan
if np.isfinite(chat_base_mu):
    ax.axhline(y=chat_base_mu, color=chat_color, linestyle='--', linewidth=1.5, alpha=0.9)

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=13)
ax.set_ylabel(r"Mean TV distance vs. Uniform", fontsize=16)
ax.set_ylim(0, 1.0)
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

handles = [Patch(facecolor=gem_color, label='Gemini')]
if any(has_chat):
    handles.append(Patch(facecolor=chat_color, label='ChatGPT'))
ax.legend(handles=handles, frameon=False, ncol=2 if any(has_chat) else 1, fontsize=12, loc='upper right')

plt.tight_layout()
fig.savefig(OUT_PDF, format="pdf", transparent=True)
fig.savefig(OUT_PNG, format="png", transparent=True)
print(f"Saved: {OUT_PDF} and {OUT_PNG}")
plt.show()

# Quick text summary
print("\n=== TV to Uniform (mean ± std across clusters) — Gemini ===")
for cond in labels:
    mu, sd = gem_stats.get(cond, (0.0, 0.0))
    print(f"  • {cond:<18} {mu:.3f} ± {sd:.3f}")
if chat_stats:
    print("\n=== TV to Uniform (mean ± std across clusters) — ChatGPT ===")
    for cond in labels:
        if cond in chat_stats:
            mu, sd = chat_stats[cond]
            print(f"  • {cond:<18} {mu:.3f} ± {sd:.3f}")