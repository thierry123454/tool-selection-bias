#!/usr/bin/env python3
import json, os, unicodedata, re, random
from collections import defaultdict

# ─── CONFIG ────────────────────────────────────────────────────────────────
CLUSTERS_PATH      = "../2_generate_clusters_and_refine/duplicate_api_clusters.json"
CLUSTER_QUERIES    = "../3_generate_queries_for_clusters/cluster_queries.json"
ORIGINAL_QUERIES   = "../data/instruction/G1_query.json"
ORIGINAL_QUERIES_2 = "../data/instruction/G2_query.json"
ORIGINAL_QUERIES_3 = "../data/instruction/G3_query.json"
TOOLENV_ROOT       = "../data/toolenv/tools"
OUTPUT_PATH        = "api_subset_selection_dataset.json"
GROUND_TRUTH_PATH  = "api_subset_selection_ground_truth.json"
SEED               = 1234
TARGETS = {2: 500, 3: 500, 4: 500, 5: 500}
NUM_CANDIDATES = 8
# ──────────────────────────────────────────────────────────────────────────

def load_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_api_definitions(path, api_map):
    """
    From the full ToolBench queries JSON, build a mapping
    from (tool_name, api_name) to the API definition dict.
    """
    queries = load_file(path)
    for q in queries:
        for api in q.get("api_list", []):
            key = (api["tool_name"], api["api_name"])
            api_map[key] = api
    return api_map

def slugify(text):
    """
    Lowercase, remove accents, replace non-alphanumeric with underscores.
    """
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text

def load_from_toolenv(category, tool, api_name):
    """
    When no definition found in api_map, attempt to load from
    data/toolenv/tools/{category}/{slugified_tool}.json -> "api_list" entry.
    Returns a definition dict or raises KeyError if not found.
    """
    # Build path to tool JSON
    category_dir = category.replace(" ", "_")
    tool_slug    = slugify(tool)
    tool_path    = os.path.join(TOOLENV_ROOT, category_dir, f"{tool_slug}.json")

    if not os.path.isfile(tool_path):
        raise KeyError(f"No tool JSON at expected path: {tool_path}")

    with open(tool_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for api in data.get("api_list", []):
        # some toolenv files use "name" instead of "api_name"
        name_field = api.get("name") or api.get("api_name")
        if name_field == api_name:
            # Build a ToolBench‐style API definition
            definition = {
                "category_name": category,
                "tool_name":     tool,
                "api_name":      api_name,
                "api_description": api.get("description", ""),
                "required_parameters": api.get("required_parameters", []),
                "optional_parameters": api.get("optional_parameters", []),
                "method": api.get("method", "GET")
            }
            # If template_response exists, include it; otherwise ignore
            if "template_response" in api:
                definition["template_response"] = api["template_response"]
            return definition

    raise KeyError(f"API '{api_name}' not found in {tool_path}")

def main():
    random.seed(SEED)

    clusters    = load_file(CLUSTERS_PATH)
    cq          = load_file(CLUSTER_QUERIES)
    api_map     = {}
    api_map     = load_api_definitions(ORIGINAL_QUERIES, api_map)
    api_map     = load_api_definitions(ORIGINAL_QUERIES_2, api_map)
    api_map     = load_api_definitions(ORIGINAL_QUERIES_3, api_map)

    all_eps = []
    all_eps_flat = []
    for cl in clusters:
        for ep in cl:
            all_eps.append((ep["tool"], ep["api_name"]))
            all_eps_flat.append(ep)

    output = []
    gt_output = []
    qid = 1

    for entry in cq:
        if qid > 2000:
            break

        K = 0
        if   qid <=  250: K = 2
        elif qid <=  500: K = 3
        elif qid <=  750: K = 4
        else:             K = 5

        cid     = entry["cluster_id"]
        queries = entry["queries"]
        cluster = clusters[cid - 1]

        # —    DEBUG: check how many queries this cluster actually has
        print(f"Cluster {cid} has {len(queries)} queries (expected 100)")

        print(f"\nProcessing Cluster {cid} (size={len(cluster)})")
        print("Sample endpoints:")
        for ep in cluster[:3]:
            print(f"  - {ep['tool']} :: {ep['api_name']}")

        print("Sample queries:")
        for q in queries[:3]:
            print(f"  • {q}")

        cluster_eps = [(ep["tool"], ep["api_name"]) for ep in cluster]
        other_eps   = [e for e in all_eps if e not in cluster_eps]

        for query_text in queries:
            if qid > 2000:
                break

            correct     = random.sample(cluster_eps, K)
            distractors = random.sample(other_eps, 8 - K)

            relevant_apis_base = correct + distractors
            random.shuffle(relevant_apis_base)

            # Now collect the corresponding full definitions in the same permuted order
            permuted_api_list = []
            for tool, api_name in relevant_apis_base:
                definition = api_map.get((tool, api_name))
                if not definition:
                    # fallback to toolenv JSON
                    for ep in all_eps_flat:
                        if ep['api_name'] == api_name:
                            category = ep['category']
                    definition = load_from_toolenv(category, tool, api_name)
                permuted_api_list.append(definition)

            output.append({
                "api_list":       permuted_api_list,
                "query":          query_text,
                "relevant APIs":  relevant_apis_base,
                "query_id":       qid
            })
            
            print(query_text)
            print(relevant_apis_base)

            # ground-truth entry for this example
            correct_indices = [
                i for i, (t, a) in enumerate(relevant_apis_base)
                if (t, a) in correct
            ]

            print(correct_indices)

            gt_output.append({
                "query_id": qid,
                "query": query_text,
                "correct_apis":     [{"tool": t, "api_name": a} for (t, a) in correct],
                "distractor_apis":  [{"tool": t, "api_name": a} for (t, a) in distractors],
                "correct_indices":  correct_indices,
            })

            qid += 1

    # Write out the combined JSON.
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    with open(GROUND_TRUTH_PATH, "w", encoding="utf-8") as f:
        json.dump(gt_output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Wrote {len(output)} queries to {OUTPUT_PATH}")
    print(f"✅ Wrote {len(gt_output)} ground-truth records to {GROUND_TRUTH_PATH}")

if __name__ == "__main__":
    main()