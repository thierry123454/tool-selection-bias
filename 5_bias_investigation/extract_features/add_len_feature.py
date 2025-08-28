#!/usr/bin/env python
import json
import os

# ─── CONFIG ────────────────────────────────────────────────────────────────
META_PATH     = 'correct_api_meta.json'
FEATURES_PATH = 'final_features.json'
# ──────────────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

# load data
meta_items     = load_json(META_PATH)
feature_items  = load_json(FEATURES_PATH)

# build lookup: (cluster_id, api) -> meta record
meta_map = {
    (m['cluster_id'], m['api']): m
    for m in meta_items
}

# update each feature entry
for feat in feature_items:
    key = (feat['cluster_id'], feat['api'])
    meta = meta_map.get(key)
    if not meta:
        # no corresponding meta; skip or set to None
        feat['desc_name_length_sum'] = None
        continue

    tool_desc = meta.get('tool_desc', '') or ''
    api_desc  = meta.get('api_desc', '') or ''
    name       = meta.get('name', '')

    # sum of lengths
    total = len(tool_desc) + len(api_desc) + len(name)

    feat['desc_name_length_sum'] = total

# write out augmented features
save_json(feature_items, FEATURES_PATH)
print(f"Wrote {len(feature_items)} records to {FEATURES_PATH}")