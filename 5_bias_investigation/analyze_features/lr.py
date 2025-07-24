#!/usr/bin/env python3
import json
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

# ─── CONFIG ────────────────────────────────────────────────────────────────
FEATURES_PATH = '../extract_features/final_features_subtract_mean.json'
OUTPUT_DIR    = 'correlation_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ──────────────────────────────────────────────────────────────────────────

# 1) Load and flatten
with open(FEATURES_PATH, 'r', encoding='utf-8') as f:
    features = json.load(f)

rows = []
for feat in features:
    base = {k: feat.get(k) for k in (
        'cluster_id','api',
        'avg_similarity_tool_desc','avg_similarity_api_desc',
        'age_days','desc_name_length_sum',
        'num_params','flesch_reading_ease','positive_word_count'
    )}
    for model, rate in feat.get('selection_rate', {}).items():
        base[f'selrate_{model}'] = rate
    rows.append(base)

df = pd.DataFrame(rows)

# identify predictor columns and model‐columns
predictors = [
    'avg_similarity_tool_desc','avg_similarity_api_desc',
    'age_days','desc_name_length_sum',
    'num_params','flesch_reading_ease','positive_word_count'
]
model_cols = [c for c in df.columns if c.startswith('selrate_')]

# 2) Linear regression for each model → print R² & coefficients, plot actual vs. predicted
print("Linear regression results:\n")
for model in model_cols:
    # drop any row with missing predictors or target
    sub = df.dropna(subset=predictors + [model])
    if len(sub) < 5:
        continue

    X = sub[predictors].values
    y = sub[model].values

    reg = LinearRegression()
    reg.fit(X, y)
    r2 = reg.score(X, y)

    # print summary
    print(f"{model:20s} R² = {r2:.3f}")
    for feat, coef in zip(predictors, reg.coef_):
        print(f"    {feat:25s} coef = {coef:+.3f}")
    print()

    # plot actual vs predicted
    y_pred = reg.predict(X)
    plt.figure(figsize=(5,5))
    plt.scatter(y, y_pred, alpha=0.6)
    m, M = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
    plt.plot([m, M], [m, M], 'k--', linewidth=1)
    plt.xlabel("Actual selection rate")
    plt.ylabel("Predicted selection rate")
    plt.title(f"{model}  (R² = {r2:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{model}__linear_fit.png"))
    plt.close()