#!/usr/bin/env python3
import json
import random
import string

# ─── CONFIG ────────────────────────────────────────────────────────────────
INPUT_JSON      = "../../3_generate_queries_for_clusters/toolbench_bias_queries.json"
TOOL_ID_MAPPING = "tool_to_id.json"
OUTPUT_JSON     = "toolbench_bias_queries_rand_name.json"
# ──────────────────────────────────────────────────────────────────────────

with open(INPUT_JSON, 'r', encoding='utf-8') as f:
    data = json.load(f)

with open(TOOL_ID_MAPPING, 'r', encoding='utf-8') as f:
    tool_to_id = json.load(f)

missing = set()

def replace_tool(name):
    if name in tool_to_id:
        return tool_to_id[name]
    else:
        missing.add(name)
        return name

for record in data:
    for api in record.get("api_list", []):
        old = api.get("tool_name")
        api["tool_name"] = replace_tool(old)

    rel = record.get("relevant APIs") or record.get("relevant_apis") or record.get("relevantApis")
    if isinstance(rel, list):
        for pair in rel:
            # pair could be [tool_name, api_name]
            if len(pair) >= 1:
                pair[0] = replace_tool(pair[0])

if missing:
    print("⚠️ Warning: no ID found for these tool names:")
    for name in sorted(missing):
        print("   ", name)

with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"✅ Written updated JSON to {OUTPUT_JSON}")