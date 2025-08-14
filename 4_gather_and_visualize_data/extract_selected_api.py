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
    # "chatgpt-temp-0-1":  "../data_bias/answer_chatgpt_temp_0_2",
    # "chatgpt-temp-0-2":  "../data_bias/answer_chatgpt_temp_0_3",
    # "chatgpt-temp-1":  "../data_bias/answer_chatgpt_temp_1",
    # "chatgpt-temp-1-2":  "../data_bias/answer_chatgpt_temp_1_2",
    # "chatgpt-temp-1-3":  "../data_bias/answer_chatgpt_temp_1_3",
    # "chatgpt-temp-2":  "../data_bias/answer_chatgpt_temp_2",
    # "chatgpt-temp-2-2":  "../data_bias/answer_chatgpt_temp_2_2",
    # "chatgpt-temp-2-3":  "../data_bias/answer_chatgpt_temp_1_3",
    # "chatgpt-temp-0.5-2":  "../data_bias/answer_chatgpt_temp_0.5",
    # "chatgpt-temp-0.5-3":  "../data_bias/answer_chatgpt_temp_0.5_2",
    # "chatgpt-top-p-0.7":  "../data_bias/answer_chatgpt_top_p_0.7",
    # "chatgpt-top-p-0.9":  "../data_bias/answer_chatgpt_top_p_0.9",
    # "claude":  "../data_bias/answer_claude",
    # "gemini":  "../data_bias/answer_gemini",
    # "deepseek":  "../data_bias/answer_deepseek",
    # "toolllama":  "../data_bias/answer_toolllama"
    # "qwen-1.7b":  "../data_bias/answer_qwen",
    # "qwen-1.7b-2":  "../data_bias/answer_qwen_2",
    # "qwen-1.7b-3":  "../data_bias/answer_qwen_3",
    # "qwen-4b":  "../data_bias/answer_qwen-4b",
    # "qwen-4b-2":  "../data_bias/answer_qwen-4b_2",
    # "qwen-4b-3":  "../data_bias/answer_qwen-4b_3",
    # "qwen-8b":  "../data_bias/answer_qwen-8b",
    # "qwen-8b-2":  "../data_bias/answer_qwen-8b_2",
    # "qwen-8b-3":  "../data_bias/answer_qwen-8b_3",
    # "qwen-14b":  "../data_bias/answer_qwen-14b",
    # "qwen-14b-2":  "../data_bias/answer_qwen-14b-2",
    # "qwen-14b-3":  "../data_bias/answer_qwen-14b-3",
    # "qwen-32b":  "../data_bias/answer_qwen-32b",
    # "qwen-32b-2":  "../data_bias/answer_qwen-32b-2",
    # "qwen-32b-3":  "../data_bias/answer_qwen-32b-3",
    # "qwen-235b": "../data_bias/answer_qwen-235b"
    # "gemini-sample":  "../data_bias/answer_gemini_sample_dist",
    # "gemini-sample-temp-2":  "../data_bias/answer_gemini_sample_dist_temp_2",
    # "gemini-rand-id":  "../data_bias/answer_gemini_rand_name",
    # "gemini-shuffle-name":  "../data_bias/answer_gemini_shuffle_name",
    # "gemini-rand-id-prom":  "../data_bias/answer_gemini_rand_name_prom",
    # "gemini-rand-id-2":  "../data_bias/answer_gemini_rand_name_2",
    # "gemini-shuffle-name-2":  "../data_bias/answer_gemini_shuffle_name_2",
    # "gemini-rand-id-prom-2":  "../data_bias/answer_gemini_rand_name_prom_2",
    # "gemini-desc-param-scramble":  "../data_bias/answer_gemini_desc_param_scramble",
    # "gemini-desc-param-scramble-2":  "../data_bias/answer_gemini_desc_param_scramble_2",
    # "gemini-desc-scramble":  "../data_bias/answer_gemini_desc_scramble",
    # "gemini-param-scramble":  "../data_bias/answer_gemini_param_scramble",
    # "gemini-desc-scramble-2":  "../data_bias/answer_gemini_desc_scramble_2",
    # "gemini-param-scramble-2":  "../data_bias/answer_gemini_param_scramble_2",
    # "gemini_desc_swap": "../data_bias/answer_gemini_desc_swap",
    # "gemini_desc_swap-2": "../data_bias/answer_gemini_desc_swap_2",
    # "gemini_desc_prom": "../data_bias/answer_gemini_desc_prom",
    # "gemini_desc_prom-2": "../data_bias/answer_gemini_desc_prom_2",
    "gemini_abo": "../data_bias/answer_gemini_all_but_one_scramble",
    # "chatgpt-rand-id":  "../data_bias/answer_chatgpt_rand_name",
    # "chatgpt-rand-id-2":  "../data_bias/answer_chatgpt_rand_name_2",
    # "chatgpt-desc-param-scramble":  "../data_bias/answer_chatgpt_desc_param_scramble",
    # "chatgpt-desc-param-scramble-2":  "../data_bias/answer_chatgpt_desc_param_scramble_2"
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
            # print(f"⚠️  Missing answer file for query_id={qid}, skipping.")
            continue

        answer = load_json(ans_path)
        api_slug, tool_slug = extract_first_action(answer)
        if not api_slug:
            # print(f"⚠️  No Action found in {ans_path}, skipping.")
            continue

        # map into cluster
        cluster_id, pos = find_cluster_id(clusters, api_slug, tool_slug)
        if cluster_id is None:
            # print(f"⚠️  (QID: {qid}) Could not map ({tool_slug}, {api_slug}) to any cluster.")
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
