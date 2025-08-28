#!/usr/bin/env python
import json
from textstat import textstat
import re

# ─── CONFIG ────────────────────────────────────────────────────────────────
META_PATH     = 'correct_api_meta.json'
FEATURES_PATH = 'final_features.json'
FEATURES_MEAN = 'final_features_subtract_mean.json'

# A small lexicon of positive words
POSITIVE_WORDS = {
    'good', 'great', 'excellent', 'effective', 'efficient',
    'simple', 'reliable', 'affordable', 'powerful', 'fast',
    'easy', 'robust', 'accurate', 'ideal', 'best',
    'innovative', 'highly', 'top', 'advanced'
}
# ──────────────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def count_positive_words(text):
    tokens = re.findall(r"\b\w+\b", text.lower())
    return sum(1 for tok in tokens if tok in POSITIVE_WORDS)

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
    feat['num_params'] =  meta.get('num_parameters', '')

    # compute Flesch Reading Ease on combined descriptions
    tool_desc = meta.get('tool_desc', '')
    api_desc  = meta.get('api_desc', '')
    full_text = f"{tool_desc} {api_desc}".strip()

    if full_text:
        # returns a float score
        score = textstat.flesch_reading_ease(full_text)
    else:
        score = None

    feat['flesch_reading_ease'] = score
    
    # count positive words in tool_desc + api_desc
    feat['positive_word_count'] = count_positive_words(full_text)

save_json(feature_items, FEATURES_PATH)
print(f"Wrote {len(feature_items)} records to {FEATURES_PATH}")

# Features to normalize
feat_names = [
    'avg_similarity_tool_desc',
    'avg_similarity_api_desc',
    'age_days',
    'desc_name_length_sum',
    'num_params',
    'flesch_reading_ease',
    'positive_word_count'
]

# Compute cluster-wise sums and counts
cluster_sums = {fname: {} for fname in feat_names}
cluster_counts = {fname: {} for fname in feat_names}

for feat in feature_items:
    cid = feat['cluster_id']
    for fname in feat_names:
        val = feat.get(fname)
        if val is None:
            continue
        cluster_sums[fname].setdefault(cid, 0.0)
        cluster_counts[fname].setdefault(cid, 0)
        cluster_sums[fname][cid] += val
        cluster_counts[fname][cid] += 1

# now divide by the true count per cluster
cluster_means = {fname: {} for fname in feat_names}
for fname in feat_names:
    for cid, total in cluster_sums[fname].items():
        count = cluster_counts[fname][cid]
        cluster_means[fname][cid] = total / count

# Subtract cluster mean from each feature value
for feat in feature_items:
    cid = feat['cluster_id']
    for fname in feat_names:
        val = feat.get(fname)
        if val is None:
            continue
        mean = cluster_means[fname].get(cid, 0.0)
        feat[fname] = val - mean

save_json(feature_items, FEATURES_MEAN)
print(f"Wrote {len(feature_items)} records to {FEATURES_MEAN}")
