#!/usr/bin/env python3
import json
import random

# ─── CONFIG ────────────────────────────────────────────────────────────────
INPUT_JSON  = '../../2_generate_clusters_and_refine/duplicate_api_clusters.json'
OUTPUT_JSON = 'tool_to_shuffled_tool'
# ──────────────────────────────────────────────────────────────────────────

with open(INPUT_JSON, 'r', encoding='utf-8') as f:
    clusters = json.load(f)

mapping = {}

for i in range(1,4):
    for cluster in clusters:
        tools = [api['tool'] for api in cluster]
        shuffled = tools.copy()
        random.shuffle(shuffled)
        for orig, new in zip(tools, shuffled):
            mapping[orig] = new

    with open(OUTPUT_JSON + "_" + str(i) + ".json", 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

print(f"Wrote {len(mapping)} entries to {OUTPUT_JSON}")