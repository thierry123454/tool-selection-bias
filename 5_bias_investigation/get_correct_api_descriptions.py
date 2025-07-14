#!/usr/bin/env python
import os
import json
import unicodedata
import re

# ─── CONFIG ──────────────────────────────────────────────────────────
TOOL_ROOT   = "data/toolenv/tools"              # ← your --tool_root_dir
CLUSTER_JSON= "../2_generate_clusters_and_refine/duplicate_api_clusters.json"
STRIPPED_JS = "stripped_tool_descriptions.json"  # from your previous script
OUT_JSON    = "flattened_api_with_stripped.json"
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

def main():
    with open(STRIPPED_JS, encoding="utf-8") as f:
        stripped_map = json.load(f)

    with open(CLUSTER_JSON, encoding="utf-8") as f:
        clusters = json.load(f)

    out = []
    for cid, cluster in enumerate(clusters, start=1):
        for idx, api in enumerate(cluster):
            display_tool = api.get("tool","")
            std_name    = slugify(display_tool)
            tool_desc   = stripped_map.get(std_name, "<MISSING DESCRIPTION>")
            api_desc    = api.get("api_desc","").strip()

            out.append({
                "cluster_id": cid,
                "api":        idx,
                "tool_desc":  tool_desc,
                "api_desc":   api_desc,
            })

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(out)} records to {OUT_JSON}")

if __name__ == "__main__":
    main()