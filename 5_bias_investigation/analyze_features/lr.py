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
# sensible global sizes
plt.rcParams.update({
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})
MODEL_COLORS = {
    "Gemini":     "#4C78A8",  # blue
    "ChatGPT 3.5":    "#BC6713",  # darker orange
    "ChatGPT 4.1":    "#F58518",  # orange
    "Claude":     "#B279A2",  # purple
    "DeepSeek":   "#E45756",  # red
    "Qwen":       "#72B7B2",  # teal
    "ToolLLaMA":  "#9D755D",  # brown
}
# ──────────────────────────────────────────────────────────────────────────

# LaTeX special chars
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
    return ''.join(TEX_ESCAPES.get(ch, ch) for ch in s)

# Load and flatten
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

# Linear regression for each model -> print R^2 & coefficients, plot actual vs. predicted
print("Linear regression results:\n")
for model in model_cols:
    model_name = model.split("_")[1]
    color = MODEL_COLORS[model_name]
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
    print(f"{model:20s} R^2 = {r2:.3f}")
    for feat, coef in zip(predictors, reg.coef_):
        print(f"    {feat:25s} coef = {coef:+.3f}")
    print()

    y_pred = reg.predict(X)

    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=300)
    ax.scatter(y, y_pred, s=28, alpha=0.6, edgecolors="none", color=color)

    # 45-degree reference
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, color="0.2")

    # tidy axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    ax.set_xlabel(r"Actual selection rate", fontsize=12)
    ax.set_ylabel(r"Predicted selection rate", fontsize=12)

    # remove top/right spines for a cleaner look
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # save tight, vector format for LaTeX
    safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in model)
    out_pdf = os.path.join(OUTPUT_DIR, f"{safe_name}__linear_fit.pdf")
    out_png = os.path.join(OUTPUT_DIR, f"{safe_name}__linear_fit.png")

    fig.savefig(out_pdf, bbox_inches="tight", transparent=True)
    fig.savefig(out_png, bbox_inches="tight", dpi=300, transparent=True)
    plt.close(fig)


# Grouped bar plot of coefficients
if coef_dict:
    feature_labels = [escape_tex(predictor) for predictor in predictors]
    models = list(coef_dict.keys())
    display_names = [m.replace("selrate_", "") for m in models]
    n_feats = len(feature_labels)
    x = np.arange(n_feats)
    width = 0.8 / max(1, len(models))

    # dynamic width so labels stay readable
    fig_w = max(8, 1.5 * n_feats)
    fig, ax = plt.subplots(figsize=(fig_w, 6), dpi=300)

    # symmetrical y-limits around zero for comparability
    max_abs = max(abs(v) for m in models for v in coef_dict[m])
    ymax = 1.15 * max_abs if max_abs > 0 else 1.0
    ax.set_ylim(-ymax, ymax)

    # bars
    for i, (model, disp) in enumerate(zip(models, display_names)):
        coefs = coef_dict[model]
        color = MODEL_COLORS.get(disp, "#999999")
        ax.bar(
            x + (i - (len(models) - 1) / 2) * width,
            coefs,
            width=width,
            color=color,
            edgecolor="none",
            alpha=0.95,
            label=fr"{escape_tex(disp)} ($R^2={r2_dict[model]:.3f}$)"
        )

    # cosmetics
    ax.axhline(0, color="0.25", linewidth=1.1, linestyle="--")           # zero line
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax.set_xticks(x)
    ax.set_xticklabels(feature_labels, rotation=30, ha="right", fontsize=12)
    ax.set_ylabel("Linear regression coefficient")

    # compact top legend
    ax.legend(frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.20), fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.90])  # leave room for legend
    barpath_pdf = os.path.join(OUTPUT_DIR, "linear_coeffs_bar.pdf")
    barpath_png = os.path.join(OUTPUT_DIR, "linear_coeffs_bar.png")
    fig.savefig(barpath_pdf, transparent=True)
    fig.savefig(barpath_png, transparent=True, dpi=300)
    print(f"Saved coefficient bar plot to {barpath_pdf} and {barpath_png}")
    plt.close(fig)