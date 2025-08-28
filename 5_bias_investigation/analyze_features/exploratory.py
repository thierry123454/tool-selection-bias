#!/usr/bin/env python3
import json
import os
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# ─── CONFIG ────────────────────────────────────────────────────────────────
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
FEATURES_PATH = '../extract_features/final_features_subtract_mean.json'
OUTPUT_DIR    = 'correlation_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ──────────────────────────────────────────────────────────────────────────

# Load and flatten
with open(FEATURES_PATH, 'r', encoding='utf-8') as f:
    features = json.load(f)

# turn into DataFrame, expanding 'selection_rate' into columns
rows = []
for feat in features:
    base = {k: feat.get(k) for k in (
        'cluster_id','api',
        'avg_similarity_tool_desc','avg_similarity_api_desc',
        'age_days','desc_name_length_sum',
        'num_params','flesch_reading_ease','positive_word_count'
    )}
    # expand each model's selection rate
    for model, rate in feat.get('selection_rate', {}).items():
        base[f'selrate_{model}'] = rate
    rows.append(base)

df = pd.DataFrame(rows)

# identify predictor columns and model columns
predictors = [
    'avg_similarity_tool_desc','avg_similarity_api_desc',
    'age_days','desc_name_length_sum',
    'num_params','flesch_reading_ease','positive_word_count'
]
model_cols = [c for c in df.columns if c.startswith('selrate_')]

# Compute and print correlations
print("Predictor vs Model selection‐rate correlations:\n")
for model in model_cols:
    print(f"--- {model} ---")
    for pred in predictors:
        # drop NaNs
        sub = df[[pred, model]].dropna()
        if len(sub) < 2:
            continue
        pearson_r, p_p = pearsonr(sub[pred], sub[model])
        spearman_r, p_s = spearmanr(sub[pred], sub[model])
        print(f"{pred:25s}  "
              f"Pearson: {pearson_r:+.3f} (p={p_p:.3f}),  "
              f"Spearman: {spearman_r:+.3f} (p={p_s:.3f})")
    print()

# Scatter plots
for model in model_cols:
    for pred in predictors:
        sub = df[[pred, model]].dropna()
        if sub.shape[0] < 2:
            continue
        plt.figure(figsize=(5,4))
        plt.scatter(sub[pred], sub[model], alpha=0.6)
        plt.xlabel(pred.replace('_',' '))
        plt.ylabel(model.replace('selrate_','selection rate '))
        plt.title(f"{model} vs {pred}")
        plt.tight_layout()
        outfn = os.path.join(OUTPUT_DIR, f"{model}__{pred}.png")
        plt.savefig(outfn)
        plt.close()