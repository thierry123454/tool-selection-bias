#!/usr/bin/env python3
import json
import random

# ─── CONFIG ────────────────────────────────────────────────────────────
INPUT_JSON  = '../../2_generate_clusters_and_refine/duplicate_api_clusters.json'
OUTPUT_JSON = 'heldout_tools.json'
# ────────────────────────────────────────────────────────────────────────

random.seed(42)  # for reproducibility

with open(INPUT_JSON, 'r', encoding='utf-8') as f:
    clusters = json.load(f)

heldout = []
for cluster in clusters:
    tools = [api.get('tool') for api in cluster if api.get('tool')]
    if not tools:
        heldout.append("")  # or could skip / use None if preferred
    else:
        heldout.append(random.choice(tools))

with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(heldout, f, indent=2, ensure_ascii=False)

print(f"Wrote {len(heldout)} held-out tools to {OUTPUT_JSON}")