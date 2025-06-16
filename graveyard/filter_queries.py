import json

# ─── CONFIG ────────────────────────────────────────────────────────────────
QUERY_FILE     = "./data/instruction/G1_query.json"   # full ToolBench query list
CLUSTERS_FILE  = "duplicate_api_clusters.json"        # generated duplicate‐clusters JSON
OUTPUT_FILE    = "filtered_queries.json"              # where to write the filtered list
# ────────────────────────────────────────────────────────────────────────────

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_cluster_endpoints(clusters):
    """
    From a list of clusters (each a list of {tool, api_name, …}),
    build a set of (tool, api_name) tuples.
    """
    s = set()
    for cluster in clusters:
        for api in cluster:
            tool = api.get("tool")
            name = api.get("api_name")
            if tool and name:
                s.add((tool, name))
    return s

def query_uses_clustered_api(q, endpoint_set):
    """
    q is a single query dict from the ToolBench JSON.
    It has a field "relevant APIs" which is a list of [tool, api_name] pairs.
    Return True if any of those appears in endpoint_set.
    """
    for rel in q.get("relevant APIs", []):
        if (rel[0], rel[1]) in endpoint_set:
            return True
    return False

def main():
    queries  = load_json(QUERY_FILE)
    clusters = load_json(CLUSTERS_FILE)

    endpoint_set = extract_cluster_endpoints(clusters)

    filtered = [q for q in queries if query_uses_clustered_api(q, endpoint_set)]

    print(f"Kept {len(filtered)} of {len(queries)} queries that reference clustered APIs.")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()