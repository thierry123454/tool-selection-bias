#!/usr/bin/env python3
import os
import json
import re
import unicodedata

# ─── CONFIG ─────────────────────────────────────────────
QUERIES_JSON    = "../data_bias/instruction/toolbench_bias_queries.json"
# QUERIES_JSON    = "../data_bias/instruction/toolbench_bias_queries_none.json"
CLUSTERS_JSON   = "../2_generate_clusters_and_refine/duplicate_api_clusters.json"
OUTPUT_PATH     = "api_selection_stats.json"

# mapping of model -> their answer directory
ANSWER_DIRS = {
    # "chatgpt_base": "../data_bias/answer_chatgpt_base",
    # "chatgpt_no_func": "../data_bias/answer_chatgpt_no_func_base_prompt",
    # "chatgpt_adj": "../data_bias/answer_chatgpt_adjusted",
    # "chatgpt_sim": "../data_bias/answer_chatgpt_similar",
    # "chatgpt_random":  "../data_bias/answer_chatgpt_base_random",
    # "chatgpt_4":  "../data_bias/answer_chatgpt_4_base_prompt",
    # "chatgpt-temp-0":  "../data_bias/answer_chatgpt_temp_0",
    # "chatgpt-temp-1":  "../data_bias/answer_chatgpt_temp_1",
    # "chatgpt-top-p-0.7":  "../data_bias/answer_chatgpt_top_p_0.7",
    # "chatgpt-top-p-0.9":  "../data_bias/answer_chatgpt_top_p_0.9",
    # "claude":  "../data_bias/answer_claude",
    # "gemini":  "../data_bias/answer_gemini",
    # "deepseek":  "../data_bias/answer_deepseek",
    # "toolllama":  "../data_bias/answer_toolllama",
    # "qwen-1.7b":  "../data_bias/answer_qwen",
    # "qwen-4b":  "../data_bias/answer_qwen-4b",
    # "qwen-8b":  "../data_bias/answer_qwen-8b"
    # "gemini-sample":  "../data_bias/answer_gemini_sample_dist",
    # "gemini-sample-temp-2":  "../data_bias/answer_gemini_sample_dist_temp_2",
    "gemini-rand-id":  "../data_bias/answer_gemini_rand_name"
}
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

def collect_stats_for_model(model_name, answers_dir, queries, clusters):
    stats = []
    for q in queries:
        qid = q["query_id"]
        ans_path = os.path.join(answers_dir, f"{qid}_CoT@1.json")
        if not os.path.isfile(ans_path):
            print(f"⚠️  Missing answer file for query_id={qid}, skipping.")
            continue

        answer = load_json(ans_path)
        api_slug, tool_slug = extract_first_action(answer)
        if not api_slug:
            print(f"⚠️  No Action found in {ans_path}, skipping.")
            continue

        # map into cluster
        cluster_id, pos = find_cluster_id(clusters, api_slug, tool_slug)
        if cluster_id is None:
            print(f"⚠️  (QID: {qid}) Could not map ({tool_slug}, {api_slug}) to any cluster.")
            continue

        # find index in the original relevant API list
        rel_list = q.get("relevant APIs", [])
        selected_idx = None
        for idx, (_, api_name) in enumerate(rel_list, start=1):
            if api_slug in slugify(api_name):
                selected_idx = idx
                break

        if selected_idx is None:
            print(f"⚠️ Slug '{api_slug}' not found in relevant APIs for qid={qid}.")
            continue

        stats.append((qid,cluster_id,pos,selected_idx))
    return stats

def main():
    queries  = load_json(QUERIES_JSON)
    clusters = load_json(CLUSTERS_JSON)

    for model_name, answers_dir in ANSWER_DIRS.items():
        print(f"\n🔍 Processing model: {model_name}")
        model_stats = collect_stats_for_model(model_name, answers_dir, queries, clusters)
        out_path = f"api_selection_stats_{model_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(model_stats, f, indent=2, ensure_ascii=False)
        print(f"✅ Wrote {len(model_stats)} records to {out_path}")

if __name__ == "__main__":
    main()
