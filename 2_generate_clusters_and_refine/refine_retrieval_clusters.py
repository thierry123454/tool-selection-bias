import json
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
import os

# ─── CONFIG ────────────────────────────────────────────────────────────
openai.api_key = os.getenv("OPENAI_API_KEY")

API_META_PATH = "../1_endpoint_metadata_and_embed/api_metadata.json"
EMBED_PATH    = "../1_endpoint_metadata_and_embed/embeddings_combined_openai.npy"
CLUSTERS_PATH   = "duplicate_api_clusters.json"

# how many neighbors to fetch for suggestion
NEIGHBOR_K      = 20
# how many embeddings to show the initial summary
DISPLAY_K       = 20
EMBED_MODEL     = "text-embedding-ada-002"
# ────────────────────────────────────────────────────────────────────────

def load_records_and_texts(meta_path):
    """Flatten api_metadata.json into a list of texts and record dicts."""
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records, texts = [], []
    for category, tools in data.items():
        for tool, info in tools.items():
            tool_desc = info.get("tool_desc", "").strip() or tool
            for api_name, api_desc in info.get("apis", []):
                txt = f"{tool_desc} | {api_name}: {api_desc}"
                records.append({
                    "category":  category,
                    "tool":      tool,
                    "tool_desc": tool_desc,
                    "api_name":  api_name,
                    "api_desc":  api_desc
                })
                texts.append(txt)
    return records, texts

def embed_query(text, model=EMBED_MODEL):
    """Call OpenAI to embed a single text."""
    resp = openai.Embedding.create(model=model, input=[text])
    return np.array(resp["data"][0]["embedding"], dtype=np.float32).reshape(1, -1)

def main():
    # load clusters, metadata, embeddings
    with open(CLUSTERS_PATH, "r", encoding="utf-8") as f:
        clusters = json.load(f)
    records, _ = load_records_and_texts(API_META_PATH)
    embs = np.load(EMBED_PATH)

    print(f"Loaded {len(clusters)} clusters, embeddings shape {embs.shape}.\n")

    # walk through all clusters of size < 5
    for ci, cluster in enumerate(clusters):
        if len(cluster) >= 5:
            continue

        print(f"\n=== Cluster {ci} (size={len(cluster)}) ===")
        for item in cluster:
            print(f"  • {item['tool']}::{item['api_name']}::{item['api_desc']}")

        # build query from first endpoint to find neighbors
        seed = cluster[0]
        q_txt = f"{seed['tool']}: {seed['tool_desc']} | {seed['api_name']}: {seed['api_desc']}"
        print(q_txt)
        q_emb = embed_query(q_txt)
        sims = cosine_similarity(q_emb, embs)[0]

        # gather top NEIGHBOR_K unique tools
        seen_tools, neighbors = set(), []
        for idx in np.argsort(sims)[::-1]:
            rec = records[idx]
            t = rec["tool"]
            if t in seen_tools or any((t == e["tool"] and rec["api_name"] == e["api_name"]) for e in cluster):
                continue
            seen_tools.add(t)
            neighbors.append(rec)
            if len(neighbors) >= NEIGHBOR_K:
                break

        # display top few suggestions
        print("\nSuggested neighbors:")
        for j, rec in enumerate(neighbors[:DISPLAY_K], 1):
            print(f" [{j}] {rec['tool']}::{rec['api_name']} — {rec['api_desc'].splitlines()[0]}")

        # interactive selection
        print("\nEnter the numbers (comma-sep) of neighbors to ADD to this cluster, or blank to skip:")
        sel = input("  add> ").strip()
        if sel:
            try:
                picks = [int(x) for x in sel.split(",") if x.strip()]
                for p in picks:
                    if 1 <= p <= len(neighbors):
                        cluster.append(neighbors[p-1])
                print(f"  → new cluster size: {len(cluster)}")
            except ValueError:
                print("  ✗ invalid input, skipping additions.")

        # optional: pause briefly
        time.sleep(0.2)

    # write updated clusters back
    with open(CLUSTERS_PATH, "w", encoding="utf-8") as f:
        json.dump(clusters, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Updated clusters written to {CLUSTERS_PATH}")

if __name__ == "__main__":
    main()
