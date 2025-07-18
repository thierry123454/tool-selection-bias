#!/usr/bin/env python
import json
import statistics
import matplotlib.pyplot as plt

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

# initialize storage
stats = {field: [] for field in numeric_fields}
# and for selection rates
model_names = set()
for feat in features:
    model_names.update(feat.get('selection_rate', {}).keys())
for model in model_names:
    stats[f'sel_rate_{model}'] = []

# collect values
for feat in features:
    for field in numeric_fields:
        stats[field].append(feat.get(field))
    for model in model_names:
        # default to 0 if missing
        sel = feat.get('selection_rate', {}).get(model, 0)
        stats[f'sel_rate_{model}'].append(sel)

# compute and print
for key, vals in stats.items():
    mean, stdev = compute_stats(vals)
    print(f"{key}: mean={mean:.4f}, stdev={stdev:.4f}")

# prepare y-values for DeepSeek
y = [feat['selection_rate'].get('DeepSeek', 0) for feat in features]

# scatter plots
for field in numeric_fields:
    x = [feat.get(field, 0) for feat in features]
    plt.figure()
    plt.scatter(x, y)
    plt.xlabel(field)
    plt.ylabel('sel_rate_DeepSeek')
    plt.title(f"{field} vs sel_rate_DeepSeek")
    plt.tight_layout()
    plt.show()