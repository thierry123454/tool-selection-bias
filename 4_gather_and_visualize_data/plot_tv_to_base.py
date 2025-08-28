import json
import math
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
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

OUTPUT_PDF = "tv_vs_base_by_perturbation.pdf"
OUTPUT_PNG = "tv_vs_base_by_perturbation.png"
TITLE = r"Impact of Metadata Perturbations on Selection (Mean TV vs.\ Base)"

K = 5
# ────────────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_rates_from_stats(stats):
    """
    Given one stats list (entries: [query_id, cluster_id, pos_in_cluster, pos_in_relevant_list]),
    return selection rates over APIs.
    """
    counts = defaultdict(lambda: defaultdict(int))
    for _, cid, pos, _ in stats:
        counts[cid][pos] += 1
    rates = {}
    for cid in counts:
        total = sum(counts[cid].values())
        rates[cid] = {i: counts[cid][i] / total for i in counts[cid]} if total else {}
    return rates

def tv_distance(dist_a, dist_b):
    """Total variation distance between two discrete distributions over 1..K."""
    return 0.5 * sum(abs(dist_a.get(i, 0.0) - dist_b.get(i, 0.0)) for i in range(1, K+1))

def average_dists_across_runs(runs, clusters):
    """
    Average selection distributions across runs for each cluster.
    runs: list of dicts {cid -> {pos -> rate}}
    returns: {cid -> {pos -> mean_rate}}
    """
    mean_per_cluster = {}
    for cid, cluster in enumerate(clusters, start=1):
        if (len(runs[0].get(cid, {})) == 0):
            continue

        K = len(cluster)
        pos_means = {}
        for pos in range(1, K + 1):
            vals = [r.get(cid, {}).get(pos, 0.0) for r in runs]
            pos_means[pos] = float(np.mean(vals)) if vals else 0.0
        mean_per_cluster[cid] = pos_means
    return mean_per_cluster

def compute_mean_std_vs_base(paths_dict, clusters):
    """
    For a given model's paths dict:
      1) average Base distribution per cluster across Base runs,
      2) compute TV vs. Base for every (condition, run, cluster),
      3) return {cond: (mean, std)} excluding "Base".
    """
    # normalize
    paths_dict = {k: (v if isinstance(v, list) else [v]) for k, v in paths_dict.items()}

    # build per-run rates
    condition_runs = {}
    for cond, paths in paths_dict.items():
        runs = []
        for p in paths:
            stats = load_json(p)
            runs.append(compute_rates_from_stats(stats))
        condition_runs[cond] = runs

    # base mean per cluster
    base_runs = condition_runs["Base"]
    base_mean = average_dists_across_runs(base_runs, clusters)

    # TV vs base per condition
    stats = {}
    for cond, runs in condition_runs.items():
        if cond == "Base":
            continue
        cond_mean = average_dists_across_runs(runs, clusters)
        tvs = []
        for cid, _ in enumerate(clusters, start=1):
            if len(cond_mean.get(cid, {})) == 0:
                continue
            tv = tv_distance(cond_mean.get(cid, {}), base_mean.get(cid, {}))
            tvs.append(tv)
        mean_tv = float(np.mean(tvs)) if tvs else 0.0
        std_tv  = float(np.std(tvs, ddof=1)) if len(tvs) > 1 else 0.0
        stats[cond] = (mean_tv, std_tv)

        print(cond)
        print(len(tvs))
        

    return stats

# Load clusters
clusters = load_json(CLUSTERS_JSON)

# Compute mean/std for both models
gem_stats  = compute_mean_std_vs_base(STATS_PATHS_GEMINI, clusters)
chat_stats = compute_mean_std_vs_base(STATS_PATHS_CHATGPT, clusters)

# Use Gemini’s order (minus Base) and include everything from Gemini
order = [k for k in STATS_PATHS_GEMINI.keys() if k != "Base"]
labels = order

# Convenience lookups
def get(g, k, idx):
    return g[k][idx] if (k in g and idx < len(g[k])) else None

# Build arrays (Gemini always present, ChatGPT may be missing)
gem_vals = [gem_stats[l][0] for l in labels]
gem_errs = [gem_stats[l][1] for l in labels]
has_chat = [l in chat_stats for l in labels]
chat_vals = [chat_stats[l][0] if l in chat_stats else None for l in labels]
chat_errs = [chat_stats[l][1] if l in chat_stats else None for l in labels]

sorted_idx = np.argsort(gem_vals)[::-1]
labels   = [labels[i]   for i in sorted_idx]
gem_vals = [gem_vals[i] for i in sorted_idx]
gem_errs = [gem_errs[i] for i in sorted_idx]
has_chat = [has_chat[i] for i in sorted_idx]
chat_vals = [chat_vals[i] for i in sorted_idx]
chat_errs = [chat_errs[i] for i in sorted_idx]

# Plot: side-by-side if ChatGPT exists; otherwise single centered Gemini bar
gem_color  = "#4C78A8"  # Gemini
chat_color = "#F58518"  # ChatGPT

fig, ax = plt.subplots(figsize=(max(7, 1.4*len(labels)), 5.0))
x = np.arange(len(labels))
w = 0.38

for i, lab in enumerate(labels):
    if has_chat[i]:
        ax.bar(x[i] - w/2, gem_vals[i], width=w, yerr=gem_errs[i],
               capsize=4, color=gem_color)
        ax.bar(x[i] + w/2, chat_vals[i], width=w, yerr=chat_errs[i],
               capsize=4, color=chat_color)
    else:
        ax.bar(x[i], gem_vals[i], width=w, yerr=gem_errs[i],
               capsize=4, color=gem_color)

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=14)
ax.set_ylabel(r"Mean TV distance vs.\ Base", fontsize=20)
# ax.set_title(TITLE)
ax.set_ylim(0, 1.0)
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

# Legend with patches
handles = [Patch(facecolor=gem_color, label='Gemini')]
if any(has_chat):
    handles.append(Patch(facecolor=chat_color, label='ChatGPT'))
ax.legend(handles=handles, frameon=False, ncol=2 if any(has_chat) else 1, fontsize=14)

plt.tight_layout()
fig.savefig(OUTPUT_PDF, format="pdf", transparent=True)
fig.savefig(OUTPUT_PNG, format="png", transparent=True)
print(f"Saved: {OUTPUT_PDF} and {OUTPUT_PNG}")
plt.show()


# --- Text summary ---------------------------------------------------------
def compute_runwise_mean_std(paths_dict, clusters):
    """
    For each non-base condition:
      - For every run: compute TV vs Base-mean per cluster, then average across clusters.
      - Return mean and std (across runs) of those per-run averages.
    """
    # normalize to lists
    paths_dict = {k: (v if isinstance(v, list) else [v]) for k, v in paths_dict.items()}

    # per-run rates for each condition
    condition_runs = {}
    for cond, paths in paths_dict.items():
        runs = []
        for p in paths:
            stats = load_json(p)
            runs.append(compute_rates_from_stats(stats))
        condition_runs[cond] = runs

    # Base mean per cluster (average over base runs)
    base_runs = condition_runs["Base"]
    base_mean = average_dists_across_runs(base_runs, clusters)

    out = {}
    for cond, runs in condition_runs.items():
        if cond == "Base":
            continue
        per_run_means = []
        for run_rates in runs:
            cluster_tvs = []
            for cid, _ in enumerate(clusters, start=1):
                if cid not in run_rates or cid not in base_mean:
                    continue
                tv = 0.5 * sum(
                    abs(run_rates[cid].get(pos, 0.0) - base_mean[cid].get(pos, 0.0))
                    for pos in range(1, K + 1)
                )
                cluster_tvs.append(tv)
            if cluster_tvs:
                per_run_means.append(float(np.mean(cluster_tvs)))
        mu = float(np.mean(per_run_means)) if per_run_means else 0.0
        sd = float(np.std(per_run_means, ddof=1)) if len(per_run_means) > 1 else 0.0
        out[cond] = (mu, sd, len(per_run_means))
    return out

def print_runwise_summary(tag, stats_dict):
    if not stats_dict:
        print(f"\nNo run-wise stats for {tag}.")
        return
    print(f"\nRun-wise mean TV ± std — {tag}:")
    max_cond, max_sd = None, -1.0
    for cond, (mu, sd, n) in stats_dict.items():
        print(f"  • {cond:<18} {mu:.3f} ± {sd:.3f}  (n={n})")
        if sd > max_sd:
            max_sd, max_cond = sd, cond
    print(f"Max run-to-run std: {max_sd:.3f} ({max_cond})")

def fmt(mu, sd): 
    return f"{mu:.3f} ± {sd:.3f}"

# Sort Gemini perturbations by impact (mean TV)
gem_sorted = sorted(
    [(cond, gem_stats[cond][0], gem_stats[cond][1]) for cond in gem_stats.keys()],
    key=lambda x: x[1],
    reverse=True
)

print("\n=== Summary: TV distance vs Base ===")
print("\nTop-8 most impactful perturbations for Gemini (mean ± std):")
for cond, mu, sd in gem_sorted[:8]:
    print(f"  • {cond:<18} {fmt(mu, sd)}")

# Model-wise overall averages (excluding Base)
gem_overall = np.mean([v[0] for k, v in gem_stats.items()])
chat_overall = np.mean([v[0] for k, v in chat_stats.items()]) if chat_stats else float('nan')
print(f"\nOverall mean TV — Gemini: {gem_overall:.3f}")
if not np.isnan(chat_overall):
    print(f"Overall mean TV — ChatGPT: {chat_overall:.3f}")

# Overlapping perturbations: side-by-side comparison
overlap = sorted(set(gem_stats.keys()).intersection(chat_stats.keys()))
if overlap:
    print("\nOverlapping perturbations (Gemini vs ChatGPT):")
    for cond in overlap:
        g_mu, g_sd = gem_stats[cond]
        c_mu, c_sd = chat_stats[cond]
        delta = g_mu - c_mu
        print(f"  • {cond:<18} {fmt(g_mu,g_sd)}  |  {fmt(c_mu,c_sd)}   Δ(G−C)={delta:+.3f}")

    # Largest gap between models
    gap_cond = max(overlap, key=lambda k: abs(gem_stats[k][0] - chat_stats[k][0]))
    gap_val = gem_stats[gap_cond][0] - chat_stats[gap_cond][0]
    print(f"\nLargest Gemini–ChatGPT gap: {gap_cond}  (Δ={gap_val:+.3f})")
else:
    print("\nNo overlapping perturbations to compare between Gemini and ChatGPT.")

gem_runwise  = compute_runwise_mean_std(STATS_PATHS_GEMINI, clusters)
chat_runwise = compute_runwise_mean_std(STATS_PATHS_CHATGPT, clusters)

print_runwise_summary("Gemini",  gem_runwise)
print_runwise_summary("ChatGPT", chat_runwise)