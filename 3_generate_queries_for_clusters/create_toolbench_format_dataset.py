#!/usr/bin/env python3
import json
import os
import unicodedata
import re

# ─── CONFIG ────────────────────────────────────────────────────────────────
CLUSTERS_PATH    = "../2_generate_clusters_and_refine/duplicate_api_clusters.json"
CLUSTER_QUERIES  = "cluster_queries.json"
ORIGINAL_QUERIES = "../data/instruction/G1_query.json"
ORIGINAL_QUERIES_2 = "../data/instruction/G2_query.json"
ORIGINAL_QUERIES_3 = "../data/instruction/G3_query.json"
TOOLENV_ROOT       = "../data/toolenv/tools"
OUTPUT_PATH      = "toolbench_bias_queries.json"
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

def slugify(text: str) -> str:
    """
    Lowercase, remove accents, replace non-alphanumeric with underscores.
    """
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text

def load_from_toolenv(category: str, tool: str, api_name: str):
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
    cq          = load_file(CLUSTER_QUERIES)
    api_map     = {}
    api_map     = load_api_definitions(ORIGINAL_QUERIES, api_map)
    api_map     = load_api_definitions(ORIGINAL_QUERIES_2, api_map)
    api_map     = load_api_definitions(ORIGINAL_QUERIES_3, api_map)

    output = []
    qid = 1

    for entry in cq:
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

        # Build the list of (tool, api_name) pairs in the cluster (in stable order)
        relevant_apis_base = [[ep["tool"], ep["api_name"]] for ep in cluster]   

        # For each query, create one entry per endpoint in the cluster,
        # so that each endpoint appears once in the first position
        for query_text in queries:
            for i in range(len(relevant_apis_base)):
                # Place endpoint i at the front; keep the others in the same relative order
                permuted_relevant = (
                    relevant_apis_base[i:] +
                    relevant_apis_base[:i]
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

    # Write out the combined JSON.
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Wrote {len(output)} queries to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()