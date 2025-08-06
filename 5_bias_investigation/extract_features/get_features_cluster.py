#!/usr/bin/env python
import os
import json
import unicodedata
import re

# ─── CONFIG ──────────────────────────────────────────────────────────
TOOL_ROOT   = "data/toolenv/tools"
CLUSTER_JSON= "../../2_generate_clusters_and_refine/duplicate_api_clusters.json"
STRIPPED_JS = "tool_metadata.json"
OUT_JSON    = "correct_api_meta.json"
BIAS_QUERIES_JSON = "../../3_generate_queries_for_clusters/toolbench_bias_queries.json"
# ─────────────────────────────────────────────────────────────────────

def slugify(text: str) -> str:
    """
    Lowercase, remove accents, replace non-alphanumeric with underscores.
    """
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text

with open(STRIPPED_JS, encoding="utf-8") as f:
    tool_map = json.load(f)

with open(CLUSTER_JSON, encoding="utf-8") as f:
    clusters = json.load(f)

param_map = {}
with open(BIAS_QUERIES_JSON, encoding="utf-8") as f:
    queries = json.load(f)
    for q in queries:
        for api in q.get("api_list", []):
            key = (api["tool_name"], api["api_name"])
            req = api.get("required_parameters", [])
            opt = api.get("optional_parameters", [])
            param_map[key] = len(req) + len(opt)

out = []
for cid, cluster in enumerate(clusters, start=1):
    for idx, api in enumerate(cluster):
        display_tool = api.get("tool","")
        std_name    = slugify(display_tool)
        tool_desc   = tool_map[std_name].get("description", "<MISSING DESCRIPTION>")
        tool_url   = tool_map[std_name].get("home_url", "<MISSING DESCRIPTION>")
        api_desc    = api.get("api_desc","").strip()
        api_name     = api.get("api_name", "")

        num_params = param_map.get((display_tool, api_name), 0)
        
        name_in_prompt =  slugify(api.get("api_name","")) + "_for_" + std_name 

        out.append({
            "cluster_id": cid,
            "api":        idx,
            "name": name_in_prompt,
            "tool_desc":  tool_desc,
            "api_desc":   api_desc,
            "url": tool_url,
            "num_parameters": num_params
        })

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print(f"Wrote {len(out)} records to {OUT_JSON}")
