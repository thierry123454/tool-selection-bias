#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
import os

# ─── CONFIG ─────────────────────────────────────────────────────────────
TV_RESULTS_PATH = "tv_results_vary_k.json"
OUTPUT_BASE     = "tv_vs_k"
OUTPUT_PDF      = OUTPUT_BASE + ".pdf"
OUTPUT_PNG      = OUTPUT_BASE + ".png"

plt.rc("text", usetex=True)
plt.rc("font", family="serif")

METHOD_LABELS = {
    "b2w": "Best-to-worst",
    "w2b": "Worst-to-best",
    "random": "Random subsets",
}

METHOD_COLORS = {
    "b2w": "#d62728",   # red
    "w2b": "#1f77b4",   # blue
    "random": "#2ca02c" # green
}

K_VALUES = [2, 3, 4, 5]  # we have k=5 only for random
RUNS = ["1", "2", "3"]
# ────────────────────────────────────────────────────────────────────────


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def max_tv_for_k(k: int) -> float:
    """
    Max TV to uniform for K categories when one category gets all mass:
    TV_max = 1 - 1/K
    """
    return 1.0 - 1.0 / k


def main():
    if not os.path.isfile(TV_RESULTS_PATH):
        raise FileNotFoundError(f"Could not find {TV_RESULTS_PATH}")

    tv_results = load_json(TV_RESULTS_PATH)

    # For each method, K, we collect lists of normalized TVs over runs
    # normalized_tv = tv / max_tv(K) -> fraction of max
    norm_api_tvs = {m: {k: [] for k in K_VALUES} for m in tv_results.keys()}
    norm_pos_tvs = {m: {k: [] for k in K_VALUES} for m in tv_results.keys()}

    for method, ks in tv_results.items():
        for k_str, runs in ks.items():
            k = int(k_str)
            if k not in K_VALUES:
                continue
            max_tv = max_tv_for_k(k)
            print(k,max_tv)

            for run in RUNS:
                if run not in runs:
                    continue
                entry = runs[run]
                api_tv = entry.get("api_tv", 0.0)
                pos_tv = entry.get("pos_tv", 0.0)

                if max_tv > 0:
                    norm_api = api_tv / max_tv
                    norm_pos = pos_tv / max_tv
                else:
                    norm_api = 0.0
                    norm_pos = 0.0

                norm_api_tvs[method][k].append(norm_api)
                norm_pos_tvs[method][k].append(norm_pos)

    # Compute mean and std (in percent) for plotting
    stats_api = {m: {"k": [], "mean": [], "std": []} for m in tv_results.keys()}
    stats_pos = {m: {"k": [], "mean": [], "std": []} for m in tv_results.keys()}
    stats_combined = {m: {"k": [], "mean": [], "std": []} for m in tv_results.keys()}

    for method in tv_results.keys():
        for k in K_VALUES:
            vals_api = norm_api_tvs[method][k]
            vals_pos = norm_pos_tvs[method][k]
            if len(vals_api) == 0:
                continue

            mean_api = 100.0 * float(np.mean(vals_api))
            std_api  = 100.0 * (float(np.std(vals_api, ddof=1)) if len(vals_api) > 1 else 0.0)

            mean_pos = 100.0 * float(np.mean(vals_pos))
            std_pos  = 100.0 * (float(np.std(vals_pos, ddof=1)) if len(vals_pos) > 1 else 0.0)

            combined_vals = [(vals_api[i] + vals_pos[i]) / 2.0 for i in range(len(vals_api))]
            mean_comb = 100.0 * float(np.mean(combined_vals))
            std_comb  = 100.0 * (float(np.std(combined_vals, ddof=1)) if len(combined_vals) > 1 else 0.0)

            stats_api[method]["k"].append(k)
            stats_api[method]["mean"].append(mean_api)
            stats_api[method]["std"].append(std_api)

            stats_pos[method]["k"].append(k)
            stats_pos[method]["mean"].append(mean_pos)
            stats_pos[method]["std"].append(std_pos)

            stats_combined[method]["k"].append(k)
            stats_combined[method]["mean"].append(mean_comb)
            stats_combined[method]["std"].append(std_comb)

    for stats in [stats_api, stats_pos, stats_combined]:
        stats['b2w']["k"].append(5)
        stats['b2w']["mean"].append(stats['random']["mean"][-1])
        stats['b2w']["std"].append(stats['random']["std"][-1])
        stats['w2b']["k"].append(5)
        stats['w2b']["mean"].append(stats['random']["mean"][-1])
        stats['w2b']["std"].append(stats['random']["std"][-1])
    
    print("Normalized API bias:")
    print(stats_api)

    print("Normalized positional bias:")
    print(stats_pos)

    print("Normalized combined bias")
    print(stats_combined)

    # ─── PLOTTING ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    ax_api, ax_pos, ax_comb = axes

    # API TV subplot
    for method in ["b2w", "w2b", "random"]:
        if method not in stats_api:
            continue
        ks = stats_api[method]["k"]
        means = stats_api[method]["mean"]
        stds = stats_api[method]["std"]
        if not ks:
            continue

        ax_api.errorbar(
            ks,
            means,
            yerr=stds,
            marker="o",
            capsize=4,
            linestyle="-",
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method, None),
            markerfacecolor=METHOD_COLORS.get(method, None),
            markeredgecolor="black",
        )

    ax_api.set_xlabel("Cluster size $K$", fontsize=13)
    ax_api.set_ylabel(r"normalized $\delta_{\mathrm{API}}$ (\% of max)", fontsize=13)
    ax_api.grid(True, alpha=0.3)
    ax_api.tick_params(axis="both", which="major", labelsize=11)

    # Positional TV subplot
    for method in ["b2w", "w2b", "random"]:
        if method not in stats_pos:
            continue
        ks = stats_pos[method]["k"]
        means = stats_pos[method]["mean"]
        stds = stats_pos[method]["std"]
        if not ks:
            continue

        ax_pos.errorbar(
            ks,
            means,
            yerr=stds,
            marker="o",
            capsize=4,
            linestyle="-",
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method, None),
            markerfacecolor=METHOD_COLORS.get(method, None),
            markeredgecolor="black",
        )

    # Combined TV subplot
    for method in ["b2w", "w2b", "random"]:
        ks = stats_combined[method]["k"]
        means = stats_combined[method]["mean"]
        stds = stats_combined[method]["std"]
        if not ks:
            continue

        ax_comb.errorbar(
            ks,
            means,
            yerr=stds,
            marker="o",
            capsize=4,
            linestyle="-",
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method, None),
            markerfacecolor=METHOD_COLORS.get(method, None),
            markeredgecolor="black",
        )

    ax_comb.set_xlabel("Cluster size $K$", fontsize=13)
    ax_comb.set_ylabel(r"Average (\%)", fontsize=13)
    ax_comb.grid(True, alpha=0.3)
    ax_comb.tick_params(axis="both", which="major", labelsize=11)

    ax_pos.set_xlabel("Cluster size $K$", fontsize=13)
    ax_pos.set_ylabel(r"normalized $\delta_{\mathrm{pos}}$ (\% of max)", fontsize=13)
    ax_pos.grid(True, alpha=0.3)
    ax_pos.tick_params(axis="both", which="major", labelsize=11)

    # Shared legend
    handles, labels = ax_api.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=11, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    plt.savefig(OUTPUT_PDF, format="pdf", transparent=True)
    plt.savefig(OUTPUT_PNG, format="png", transparent=True)
    print(f"Saved plots to {OUTPUT_PDF} and {OUTPUT_PNG}")

    plt.show()


if __name__ == "__main__":
    main()