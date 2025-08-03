#!/usr/bin/env python3
import json
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# ─── CONFIG ────────────────────────────────────────────────────────────────
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
FEATURES_PATH = '../extract_features/final_features_subtract_mean.json'
OUTPUT_DIR    = 'correlation_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ──────────────────────────────────────────────────────────────────────────

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

# containers for bar plot
coef_dict = {}
r2_dict = {}

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

    # store for later bar plot
    coef_dict[model] = reg.coef_.tolist()
    r2_dict[model] = r2

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
    # plt.title(f"{model}  ($R^2$ = {r2:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{model}__linear_fit.pdf"))
    plt.close()


# 3) Grouped bar plot of coefficients
if coef_dict:
    feature_labels = [escape_tex(predictor) for predictor in predictors]
    models = list(coef_dict.keys())
    # short names without "selrate_"
    display_names = [escape_tex(m.replace("selrate_", "")) for m in models]
    n_feats = len(feature_labels)
    x = np.arange(n_feats)
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(10, 4))
    for i, model in enumerate(models):
        coefs = coef_dict[model]
        ax.bar(
            x + (i - (len(models) - 1) / 2) * width,
            coefs,
            width=width,
            label=f"{model.replace('selrate_', '')} (R²={r2_dict[model]:.3f})"
        )

    ax.set_xticks(x)
    ax.set_xticklabels(feature_labels, rotation=45, ha="right")
    ax.set_ylabel("Linear regression coefficient")
    # ax.set_title("Feature weights predicting API selection rate")
    ax.legend(frameon=False, fontsize="small")
    plt.tight_layout()
    barpath_pdf = os.path.join(OUTPUT_DIR, "linear_coeffs_bar.pdf")
    barpath_png = os.path.join(OUTPUT_DIR, "linear_coeffs_bar.png")
    fig.savefig(barpath_pdf, transparent=True)
    fig.savefig(barpath_png, transparent=True)
    print(f"Saved coefficient bar plot to {barpath_pdf} and {barpath_png}")
    plt.close()