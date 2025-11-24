#!/usr/bin/env python3
import json
import os
import unicodedata
import re
import random

# ─── CONFIG ────────────────────────────────────────────────────────────────
CLUSTERS_PATH    = "../2_generate_clusters_and_refine/duplicate_api_cluster_sentiment.json"
CLUSTER_QUERIES  = "cluster_queries.json"
ORIGINAL_QUERIES = "../data/instruction/G1_query.json"
ORIGINAL_QUERIES_2 = "../data/instruction/G2_query.json"
ORIGINAL_QUERIES_3 = "../data/instruction/G3_query.json"
TOOLENV_ROOT       = "../data/toolenv/tools"
OUTPUT_PATH = f"toolbench_bias_queries_small_k.json"
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
    clusters    = load_file(CLUSTERS_PATH)
    queries          = load_file(CLUSTER_QUERIES)[6]['queries']
    api_map     = {}
    api_map     = load_api_definitions(ORIGINAL_QUERIES, api_map)
    api_map     = load_api_definitions(ORIGINAL_QUERIES_2, api_map)
    api_map     = load_api_definitions(ORIGINAL_QUERIES_3, api_map)

    cluster = clusters[0]

    # —    DEBUG: check how many queries this cluster actually has
    print(f"Cluster has {len(queries)} queries (expected 100)")
    print("Sample endpoints:")
    for ep in cluster[:3]:
        print(f"  - {ep['tool']} :: {ep['api_name']}")

    print("Sample queries:")
    for q in queries[:3]:
        print(f"  • {q}")

    # Build the list of (tool, api_name) pairs in the cluster (in stable order)
    relevant_apis_base = [[ep["tool"], ep["api_name"]] for ep in cluster]

    print(relevant_apis_base)

    subset_sizes = [2, 3, 4]
    num_subsets_per_k = 1
    w2b = [['Sentiment Analysis_v12', 'Text Analysis'], ['TextSentAI  -  AI powered Text Sentiment Analyzer ', 'TextSentAI API 📊'], ['Multi-lingual Sentiment Analysis', 'Sentiment Analysis'], ['Sentiment by API-Ninjas', '/v1/sentiment'], ['Sentiment Analysis Service', 'Analyze Text']]
    b2w = w2b[::-1]

    print(b2w)


    # For each query, create one entry per endpoint in the cluster,
    # so that each endpoint appears once in the first position (that is, if SHUFFLE is set to cyclic)
    for method in ['w2b', 'b2w', 'random']:
        if method == 'random':
            subset_sizes.append(5)
            num_subsets_per_k = 3

        for k in subset_sizes:
            out_dir = f"k={k}"
            os.makedirs(out_dir, exist_ok=True)

            if k == 5:
                num_subsets_per_k = 1

            print(f"\n=== Generating subsets for K={k} ===")

            for subset_idx in range(num_subsets_per_k):
                if method == 'random':
                    subset_apis = random.sample(relevant_apis_base, k)
                elif method == 'w2b':
                    subset_apis = w2b[:k]
                elif method == 'b2w':
                    subset_apis = b2w[:k]

                print(f"  Subset {subset_idx + 1} for K={k}: {subset_apis}")

                output = []
                qid = 1

                for query_text in queries:
                    for i in range(len(subset_apis)):
                        # Place endpoint i at the front; keep the others in the same relative order (if SHUFFLE is not random)
                        permuted_relevant = (
                            subset_apis[i:] +
                            subset_apis[:i]
                        )

                        # Now collect the corresponding full definitions in the same permuted order
                        permuted_api_list = []
                        for tool, api_name in permuted_relevant:
                            definition = api_map.get((tool, api_name))
                            if not definition:
                                # fallback to toolenv JSON
                                for ep in cluster:
                                    if ep['api_name'] == api_name:
                                        category = ep['category']
                                definition = load_from_toolenv(category, tool, api_name)
                            permuted_api_list.append(definition)

                        output.append({
                            "api_list":       permuted_api_list,
                            "query":          query_text,
                            "relevant APIs":  permuted_relevant,
                            "query_id":       qid
                        })
                        qid += 1

                out_path = os.path.join(
                        out_dir,
                        f"toolbench_bias_queries_{method}_k{k}_subset{subset_idx + 1}.json"
                    )
                
                # Write out the combined JSON.
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(output, f, indent=2, ensure_ascii=False)

                print(f"\n✅ Wrote {len(output)} queries to {out_path}")

if __name__ == "__main__":
    main()