import json
import numpy as np
import matplotlib.pyplot as plt

# Setup LaTeX
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# ─── CONFIG ────────────────────────────────────────────────────────────
STATS_PATHS = {
    "ChatGPT": "api_selection_stats_chatgpt_no_func.json",
    "ChatGPT Func": "api_selection_stats_chatgpt_base.json",
    "Claude":  "api_selection_stats_claude.json",
    "Gemini":  "api_selection_stats_gemini.json",
    "DeepSeek":  "api_selection_stats_deepseek.json",
}
CLUSTERS_JSON  = "../2_generate_clusters_and_refine/duplicate_api_clusters.json"
# ────────────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

clusters    = load_json(CLUSTERS_JSON)
model_stats = {name: load_json(path) for name, path in STATS_PATHS.items()}

counts = {name: {} for name in model_stats}
for name, stats in model_stats.items():
    for _, cid, pos, _ in stats:
        counts[name].setdefault(cid, {}).setdefault(pos, 0)
        counts[name][cid][pos] += 1

rates = {}
for name, cd in counts.items():
    rates[name] = {}
    for cid, posdict in cd.items():
        total = sum(posdict.values())
        rates[name][cid] = {pos: cnt/total for pos, cnt in posdict.items()} if total else {}

# define all (cluster_id, position) slots
features = []
for cid, cluster in enumerate(clusters, start=1):
    for pos in range(1, len(cluster)+1):
        features.append((cid, pos))

models = list(STATS_PATHS.keys())
n = len(models)

# build one long vector per model
vectors = np.zeros((n, len(features)), dtype=float)
for i, name in enumerate(models):
    for j, (cid, pos) in enumerate(features):
        vectors[i, j] = rates[name].get(cid, {}).get(pos, 0.0)

print(vectors)

# compute Pearson correlation matrix
corr = np.corrcoef(vectors)

# plot heatmap
fig, ax = plt.subplots(figsize=(6,6))
im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
# ticks & labels
ax.set_xticks(np.arange(n))
ax.set_yticks(np.arange(n))
ax.set_xticklabels(models, rotation=45, ha="right")
ax.set_yticklabels(models)
# annotate values
for i in range(n):
    for j in range(n):
        txt = f"{corr[i,j]:.2f}"
        ax.text(j, i, txt, ha="center", va="center", color="white" if abs(corr[i,j])>0.5 else "black")
ax.set_title("Model Selection-Bias Correlation Matrix")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")


fig.savefig("model_correlation.pdf", format="pdf", transparent=True)
print(f"Saved chart grid to model_correlation.pdf")
plt.tight_layout()
plt.show()