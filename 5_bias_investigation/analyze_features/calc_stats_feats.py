#!/usr/bin/env python
import json
import statistics
import matplotlib.pyplot as plt
from collections import defaultdict

FEATURES_PATH = '../extract_features/final_features.json'

def compute_stats(values):
    """Compute mean and stdev, handling empty or single-element lists."""
    clean_vals = [v for v in values if v is not None]
    mean = statistics.mean(clean_vals)
    stdev = statistics.stdev(clean_vals)
    return mean, stdev

with open(FEATURES_PATH, 'r', encoding='utf-8') as f:
    features =  json.load(f)

# define top-level numeric fields
numeric_fields = [
    'avg_similarity_tool_desc',
    'avg_similarity_api_desc',
    'age_days',
    'desc_name_length_sum',
    'num_params',
    'flesch_reading_ease',
    'positive_word_count'
]

# find all model names for sel_rate
model_names = set()
for feat in features:
    model_names.update(feat.get('selection_rate', {}).keys())

clusters = defaultdict(lambda: defaultdict(list))

# collect values by cluster
for feat in features:
    cid = feat['cluster_id']
    for fld in numeric_fields:
        clusters[cid][fld].append(feat.get(fld))
    for m in model_names:
        clusters[cid][f'sel_rate_{m}'].append(
            feat.get('selection_rate', {}).get(m, 0)
        )

print(clusters)

# compute and print
for cid in sorted(clusters):
    print(f"\n=== Cluster {cid} ===")
    for key, vals in clusters[cid].items():
        mean, stdev = compute_stats(vals)
        if mean is None:
            print(f"{key}: no data")
        else:
            print(f"{key:25s} mean={mean:.4f}  stdev={stdev:.4f}")

# prepare y-values for DeepSeek
# y = [feat['selection_rate'].get('DeepSeek', 0) for feat in features]

# scatter plots
# for field in numeric_fields:
#     x = [feat.get(field, 0) for feat in features]
#     plt.figure()
#     plt.scatter(x, y)
#     plt.xlabel(field)
#     plt.ylabel('sel_rate_DeepSeek')
#     plt.title(f"{field} vs sel_rate_DeepSeek")
#     plt.tight_layout()
#     plt.show()