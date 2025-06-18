#!/usr/bin/env python3
import os
import json
import re
import unicodedata

# ─── CONFIG ─────────────────────────────────────────────
ANSWERS_DIR     = "./data_bias/answer_chatgpt"
QUERIES_JSON    = "./data_bias/instruction/toolbench_bias_queries.json"
CLUSTERS_JSON   = "2_generate_clusters_and_refine/duplicate_api_clusters.json"
OUTPUT_PATH     = "api_selection_stats.json"
# ────────────────────────────────────────────────────────

def slugify(text):
    """Lowercase, remove accents, replace non-alnum with underscores."""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if text and text[0].isdigit():
        text = "get_" + text
    return text

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def find_cluster_id(clusters, api_slug, tool_slug):
    """
    Return (cluster_idx, position_in_cluster)
    or (None, None) if not found.
    """
    for ci, cluster in enumerate(clusters, start=1):
        for pi, ep in enumerate(cluster, start=1):
            ep_slug_tool = slugify(ep["tool"])
            ep_slug_api = slugify(ep["api_name"])
            if api_slug in ep_slug_api and tool_slug in ep_slug_tool:
                return ci, pi
    return None, None

def extract_first_action(answer_json):
    """
    From one answer JSON, extract the first Action node description.
    Returns the tool_slug (the part after the last '_for_'), or None if not found.
    """
    try:
        trys = answer_json["trys"]
        chain = trys[0]["chain"]
        for node in chain:
            if node.get("node_type") == "Action":
                desc = node["description"]
                # expected format: "{api_slug}_for_{tool_slug}"
                if "_for_" in desc:
                    api_slug, tool_slug = desc.rsplit("_for_", 1)
                    return api_slug, tool_slug
                else:
                    # fallback: treat entire desc as tool_slug
                    return desc, None
    except Exception:
        pass
    return None, None

def main():
    queries = load_json(QUERIES_JSON)
    clusters = load_json(CLUSTERS_JSON)
    counter = 0
    stats = []
    for q in queries:
        qid = q["query_id"]
        ans_path = os.path.join(ANSWERS_DIR, f"{qid}_CoT@1.json")
        print(ans_path)
        if not os.path.isfile(ans_path):
            print(f"⚠️  Missing answer file for query_id={qid}, skipping.")
            continue

        answer = load_json(ans_path)
        api_slug, tool_slug = extract_first_action(answer)
        if not api_slug:
            print(f"⚠️  No Action found in {ans_path}, skipping.")
            continue

        # find which cluster it belongs to
        cluster_id, pos = find_cluster_id(clusters, api_slug, tool_slug)
        if cluster_id is None:
            print(f"⚠️  Could not map ({tool_slug}, {api_slug}) to any cluster.")
            continue

        # find where in the query's relevant APIs this tool was listed
        rel = q.get("relevant APIs", [])
        selected_idx = None
        for idx, (_, api_name) in enumerate(rel, start=1):
            if api_slug in slugify(api_name):
                selected_idx = idx
                break

        if selected_idx is None:
            print(f"⚠️ Slug '{api_slug}' not found in relevant APIs for qid={qid}.")
            continue

        stats.append((qid, cluster_id, pos, selected_idx))

    # write out
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Wrote {len(stats)} selection records to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
