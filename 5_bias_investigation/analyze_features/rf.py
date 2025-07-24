#!/usr/bin/env python3
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ─── CONFIG ────────────────────────────────────────────────────────────────
FEATURES_PATH = '../extract_features/final_features_subtract_mean.json'
OUTPUT_DIR    = 'importance_plots'
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
        'desc_name_length_sum',
        'num_params','flesch_reading_ease','positive_word_count'
    )}
    for model, rate in feat.get('selection_rate', {}).items():
        base[f'selrate_{model}'] = rate
    rows.append(base)

df = pd.DataFrame(rows).dropna(subset=[
    'avg_similarity_tool_desc','avg_similarity_api_desc',
    'desc_name_length_sum',
    'num_params','flesch_reading_ease','positive_word_count'
])

# predictors and targets
predictors = [
    'avg_similarity_tool_desc','avg_similarity_api_desc',
    'desc_name_length_sum',
    'num_params','flesch_reading_ease','positive_word_count'
]
model_cols = [c for c in df.columns if c.startswith('selrate_')]

for model in model_cols:
    # prepare data
    data = df[predictors + [model]].dropna()
    X = data[predictors].values
    y = data[model].values

    print(f"\n{model} performance:")
    # train Random Forest
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )

    # print(X)

    # print(len(y))

    rf.fit(X, y)

    r2s = cross_val_score(rf, X, y, cv=5, scoring="r2", n_jobs=-1)
    print(f"  5-fold CV R²: mean={r2s.mean():.3f}, std={r2s.std():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
    )

    # 2) train on train‐split
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        random_state=42,
        n_jobs=-1,
        oob_score=True    # see note below
    )
    rf.fit(X_train, y_train)

    # 3) evaluate on test‐split
    y_pred = rf.predict(X_test)
    r2    = r2_score(y_test, y_pred)
    mse   = mean_squared_error(y_test, y_pred)

    print(f"  R² on test set = {r2:.3f}")
    print(f"  MSE on test set = {mse:.3g}")

    # built-in feature importances
    imp = rf.feature_importances_
    imp_sorted_idx = np.argsort(imp)[::-1]
    # print(f"\n=== {model} Random Forest Feature Importances ===")
    # for idx in imp_sorted_idx:
    #     print(f"  {predictors[idx]:25s}: {imp[idx]:.3f}")

    # permutation importances (on the same data)
    perm = permutation_importance(rf, X, y, n_repeats=10, random_state=0, n_jobs=-1)
    perm_means = perm.importances_mean
    perm_idx = np.argsort(perm_means)[::-1]
    # print(f"\n=== {model} Permutation Importances ===")
    # for idx in perm_idx:
        # print(f"  {predictors[idx]:25s}: {perm_means[idx]:.3f}")

    # plot both
    # fig, axes = plt.subplots(1, 2, figsize=(10,4), sharey=True)
    # axes[0].barh([predictors[i] for i in imp_sorted_idx], imp[imp_sorted_idx])
    # axes[0].set_title('RF built-in importance')
    # axes[0].invert_yaxis()
    # axes[1].barh([predictors[i] for i in perm_idx], perm_means[perm_idx])
    # axes[1].set_title('Permutation importance')
    # axes[1].invert_yaxis()
    # fig.suptitle(model)
    # plt.tight_layout()
    # plt.subplots_adjust(top=0.88)
    # plt.savefig(os.path.join(OUTPUT_DIR, f"{model}_importances.png"))
    # plt.close()