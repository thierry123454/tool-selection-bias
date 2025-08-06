#!/usr/bin/env python
import json

# ─── CONFIG ────────────────────────────────────────────────────────────────
FEATURES_PATH = 'final_features.json'
RATES_PATH    = '../../4_gather_and_visualize_data/rates_api.json'
# ──────────────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

# load existing features and the selection‐rates data
features = load_json(FEATURES_PATH)
rates    = load_json(RATES_PATH)

# for each feature entry, add a `selection_rate` dict
for feat in features:
    cluster_str = str(feat['cluster_id'])
    api_str     = str(feat['api'] + 1)  # rates_api uses 1-based indexing
    sel_rates   = {}

    for model_name, clusters in rates.items():
        # default to None if missing
        sel_rates[model_name] = clusters.get(cluster_str, {}) \
                                        .get(api_str, 0)

    feat['selection_rate'] = sel_rates

# write out the augmented features
save_json(features, FEATURES_PATH)
print(f"Wrote {len(features)} records to {FEATURES_PATH}")
