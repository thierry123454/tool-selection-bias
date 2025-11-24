#!/usr/bin/env python3
import os
import json
import unicodedata
import re

# ─── CONFIG ─────────────────────────────────────────────────────────────
INSTRUCTION_ROOT = "../data_bias/instruction/"  # contains k=2, k=3, k=4 subfolders
ANSWER_ROOT      = "../data_bias/"            # base folder that has answer_qwen_* dirs
METHODS          = ["b2w", "w2b", "random"]
K_VALUES         = [2, 3, 4]
RUNS             = [1, 2, 3]
# ────────────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def slugify(text):
    """Lowercase, remove accents, replace non-alnum with underscores."""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text

def tv_to_uniform(dist):
    """
    Total variation distance between dist and the uniform distribution
    over the same support: TV = 0.5 * sum_i |p_i - 1/n|.
    Returns 0.0 if dist is empty.
    """
    n = len(dist)
    if n == 0:
        return 0.0
    u = 1.0 / n
    return 0.5 * sum(abs(p - u) for p in dist)

def extract_first_action(answer_json):
    """
    From one answer JSON, extract the first Action node description.
    Returns (api_slug, tool_slug) parsed from '..._for_...' or (None, None).
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
                    # no "_for_", fallback: treat entire desc as api_slug
                    return desc, None
    except Exception:
        pass
    return None, None

def build_endpoint_id_map(instruction_data):
    """
    Build a stable mapping (tool_name, api_name) -> api_id for a given run
    by scanning all 'relevant APIs' entries in the instruction JSON.
    """
    endpoint_to_id = {}
    next_id = 0

    for item in instruction_data:
        rel_apis = item.get("relevant APIs", [])
        for tool_name, api_name in rel_apis:
            key = (tool_name, api_name)
            if key not in endpoint_to_id:
                endpoint_to_id[key] = next_id
                next_id += 1

    return endpoint_to_id

def match_endpoint_for_query(api_slug, tool_slug, relevant_apis):
    """
    Given api_slug, tool_slug, and the 'relevant APIs' list for a query,
    return (endpoint_key, position) where endpoint_key is (tool_name, api_name)
    and position is 1-based index in relevant_apis.
    """
    if not api_slug:
        return None, None

    for idx, (tool_name, api_name) in enumerate(relevant_apis, start=1):
        api_name_slug = slugify(api_name)
        tool_name_slug = slugify(tool_name)
        # require that api_slug is contained in api_name slug;
        # if tool_slug exists, also require it's contained in tool_name slug.
        api_match = api_slug in api_name_slug
        tool_match = True if tool_slug is None else (tool_slug in tool_name_slug)
        if api_match and tool_match:
            return (tool_name, api_name), idx

    # If not found, you might want to relax matching or log a warning
    # print(f"⚠️ No match for api_slug={api_slug}, tool_slug={tool_slug}")
    return None, None

def process_run(method, k, run):
    """
    Process one (method, k, run) combo:
    - Load instruction file
    - Build endpoint_id map for that run
    - Walk over all queries, read answer JSONs, extract chosen endpoint,
      and accumulate counts.
    Returns a dict with:
      - "api_dist": normalized distribution over API ids
      - "pos_dist": normalized distribution over positions (1..k)
      - "n": total number of selections used to compute the dists
    """
    # Determine which subset index to use for the instruction file
    if method in ["b2w", "w2b"] or (method == 'random' and k == 5):
        subset_idx = 1
    else:
        subset_idx = run  # random_k{k}_subset{run}.json

    inst_path = os.path.join(
        INSTRUCTION_ROOT,
        f"k={k}",
        f"toolbench_bias_queries_{method}_k{k}_subset{subset_idx}.json"
    )

    if not os.path.isfile(inst_path):
        return {"api_dist": [], "pos_dist": [], "n": 0}

    instruction_data = load_json(inst_path)

    # Build endpoint -> id map for this run
    endpoint_to_id = build_endpoint_id_map(instruction_data)
    num_apis = len(endpoint_to_id)
    # positions are 1..k
    max_pos = k

    # Directory with answer JSON files
    answers_dir = os.path.join(
        ANSWER_ROOT,
        f"answer_qwen_{method}_k={k}_{run}"
    )
    if not os.path.isdir(answers_dir):
        return {"api_dist": [0.0] * num_apis, "pos_dist": [0.0] * max_pos, "n": 0}

    # Index instruction data by query_id for quick lookup
    query_by_id = {item["query_id"]: item for item in instruction_data}

    # Count selections
    api_counts = [0] * num_apis
    pos_counts = [0] * max_pos
    total = 0

    # Iterate over all queries in this instruction file
    for qid, item in query_by_id.items():
        ans_path = os.path.join(answers_dir, f"{qid}_CoT@1.json")
        if not os.path.isfile(ans_path):
            print(f"⚠️ Missing answer file {ans_path}, skipping.")
            continue

        answer_json = load_json(ans_path)
        api_slug, tool_slug = extract_first_action(answer_json)
        if not api_slug:
            print(f"⚠️ No Action node found in {ans_path}, skipping.")
            continue

        relevant_apis = item.get("relevant APIs", [])
        endpoint_key, pos = match_endpoint_for_query(api_slug, tool_slug, relevant_apis)
        if endpoint_key is None:
            print(f"⚠️ Could not map ({api_slug}, {tool_slug}) to relevant APIs for qid={qid}")
            continue

        api_id = endpoint_to_id.get(endpoint_key)
        if api_id is None:
            print(f"⚠️ Endpoint {endpoint_key} not in endpoint_to_id for qid={qid}")
            continue

        # pos is 1-based index in the query's relevant APIs list
        if 1 <= pos <= max_pos and 0 <= api_id < num_apis:
            api_counts[api_id] += 1
            pos_counts[pos - 1] += 1
            total += 1

    if total == 0:
        api_dist = [0.0] * num_apis
        pos_dist = [0.0] * max_pos
    else:
        api_dist = [c / total for c in api_counts]
        pos_dist = [c / total for c in pos_counts]

    return {"api_dist": api_dist, "pos_dist": pos_dist, "n": total}

def main():
    results = {}

    for method in METHODS:
        results[method] = {}
        for k in K_VALUES:
            results[method][str(k)] = {}
            for run in RUNS:
                dists = process_run(method, k, run)
                results[method][str(k)][str(run)] = dists
                print(f"{method}, k={k}, run={run}: {dists['n']} records")

    # handle random k=5
    results.setdefault('random', {})
    results['random'][str(5)] = {}
    for run in RUNS:
        dists = process_run('random', 5, run)
        results['random'][str(5)][str(run)] = dists
        print(f"random, k=5, run={run}: {dists['n']} records")
    print(results)

    tv_results = {}
    for method, ks in results.items():
        tv_results[method] = {}
        for k, runs in ks.items():
            tv_results[method][k] = {}
            for run, dists in runs.items():
                api_tv = tv_to_uniform(dists["api_dist"])
                pos_tv = tv_to_uniform(dists["pos_dist"])
                tv_results[method][k][run] = {
                    "api_tv": api_tv,
                    "pos_tv": pos_tv,
                }

    print(tv_results)

    # save if needed
    with open("tv_results_vary_k.json", "w", encoding="utf-8") as f:
        json.dump(tv_results, f, indent=2, ensure_ascii=False)
        
    print("\nDone.")
    return results

if __name__ == "__main__":
    main()