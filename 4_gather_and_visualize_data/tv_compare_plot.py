import json
from collections import defaultdict

# ─── CONFIG ────────────────────────────────────────────────────────────
# STATS_PATHS = {
#     "ChatGPT 3.5":  "selection_stats/api_selection_stats_chatgpt_base.json",
#     "ChatGPT 4.1":  "selection_stats/api_selection_stats_chatgpt_4.json",
#     "Claude":  "selection_stats/api_selection_stats_claude.json",
#     "Gemini":  "selection_stats/api_selection_stats_gemini.json",
#     "DeepSeek":  "selection_stats/api_selection_stats_deepseek.json",
#     "Qwen (32B)":  "selection_stats/api_selection_stats_qwen-32b.json",
#     "ToolLLama":  "selection_stats/api_selection_stats_toolllama.json"
# }
STATS_PATHS = {
    "ChatGPT 3.5":  "selection_stats/api_selection_stats_chatgpt_base.json",
    "ChatGPT 4.1":  "selection_stats/api_selection_stats_chatgpt_4.json",
    "Gemini":  "selection_stats/api_selection_stats_gemini.json",
    "DeepSeek":  "selection_stats/api_selection_stats_deepseek.json",
    "Qwen":  "selection_stats/api_selection_stats_qwen-235b.json",
}
CLUSTERS_JSON = "../2_generate_clusters_and_refine/duplicate_api_clusters.json"
# ────────────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# Load clusters & model‐specific stats
clusters    = load_json(CLUSTERS_JSON)
model_stats = {name: load_json(path) for name, path in STATS_PATHS.items()}

# Build counts for (a) pos_in_cluster and (b) pos_in_list
counts_api = {m: defaultdict(lambda: defaultdict(int)) for m in model_stats}
counts_pos = {m: defaultdict(lambda: defaultdict(int)) for m in model_stats}

for model, stats in model_stats.items():
    for _, cid, pos_cluster, pos_list in stats:
        counts_api[model][cid][pos_cluster] += 1
        counts_pos[model][cid][pos_list]    += 1

# Compute selection‐rate p_i = count_i / total_counts
rates_api = {m: {} for m in counts_api}
rates_pos = {m: {} for m in counts_pos}

for m in model_stats:
    # API‐bias rates
    for cid, pd in counts_api[m].items():
        total = sum(pd.values())
        rates_api[m][cid] = {i: pd[i]/total for i in pd} if total else {}
    # Position‐bias rates
    for cid, pd in counts_pos[m].items():
        total = sum(pd.values())
        rates_pos[m][cid] = {i: pd[i]/total for i in pd} if total else {}

with open('rates_api.json', 'w', encoding='utf-8') as f:
    json.dump(rates_api, f, indent=2, ensure_ascii=False)

# Compute TV‐distances & combined metric
tv_api_by_model   = {m: [] for m in STATS_PATHS}
tv_pos_by_model   = {m: [] for m in STATS_PATHS}
tv_combined_by_model = {m: [] for m in STATS_PATHS}

print("Model     Cluster │   D_api   D_pos   D_combined")
print("─" *  50)
for cid, cluster in enumerate(clusters, start=1):
    K = len(cluster)
    uniform = 1.0 / K
    for m in STATS_PATHS:
        p_api = rates_api[m].get(cid, {})
        p_pos = rates_pos[m].get(cid, {})
        # TV distance for API‐bias
        tv_api = 0.5 * sum(abs(p_api.get(i, 0.0) - uniform) for i in range(1, K+1))
        # TV distance for position‐bias
        tv_pos = 0.5 * sum(abs(p_pos.get(i, 0.0) - uniform) for i in range(1, K+1))
        # Combined
        tv_combined = (tv_api + tv_pos) / 2

        tv_api_by_model[m].append(tv_api)
        tv_pos_by_model[m].append(tv_pos)
        tv_combined_by_model[m].append(tv_combined)

        print(f"{m:8s} {cid:7d} │ {tv_api:7.3f} {tv_pos:7.3f} {tv_combined:11.3f}")
    print()

# Finally, average across clusters
print("Average over all clusters:")
for m in STATS_PATHS:
    avg_api   = sum(tv_api_by_model[m])   / len(tv_api_by_model[m])
    avg_pos   = sum(tv_pos_by_model[m])   / len(tv_pos_by_model[m])
    avg_comb  = sum(tv_combined_by_model[m]) / len(tv_combined_by_model[m])
    print(f"{m:8s}:  D_api={avg_api:5.3f},  D_pos={avg_pos:5.3f},  D_combined={avg_comb:5.3f}")