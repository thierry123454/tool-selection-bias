#!/usr/bin/env python3
import json
import random
import string

# ─── CONFIG ────────────────────────────────────────────────────────────────
INPUT_JSON = '../../2_generate_clusters_and_refine/duplicate_api_clusters.json'
OUTPUT_JSON = 'tool_to_id.json'
# ──────────────────────────────────────────────────────────────────────────

def random_id(n=20):
    alphabet = string.ascii_lowercase + string.digits
    return ''.join(random.choices(alphabet, k=n))

with open(INPUT_JSON, 'r', encoding='utf-8') as f:
    clusters = json.load(f)

tools = set()
for cluster in clusters:
    for api in cluster:
        tools.add(api['tool'])

mapping = {tool: random_id(20) for tool in sorted(tools)}

with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(mapping, f, indent=2, ensure_ascii=False)

print(f"Wrote {len(mapping)} entries to {OUTPUT_JSON}")