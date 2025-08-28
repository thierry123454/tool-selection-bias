#!/usr/bin/env python3
import os
import json
from openai import OpenAI
import numpy as np
from collections import defaultdict

# ─── CONFIG ────────────────────────────────────────────────────────────────
API_META_FILE        = "correct_api_meta.json"
CLUSTER_QUERIES_FILE = "../../3_generate_queries_for_clusters/cluster_queries.json"
OUTPUT_FILE          = "avg_similarities_embeddings.json"
EMBEDDING_MODEL      = "text-embedding-ada-002"
# ──────────────────────────────────────────────────────────────────────────

client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
    
def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def batch_embed(texts, model):
    """Call OpenAI Embeddings API once for a batch of texts."""
    resp = client.embeddings.create(input=texts, model=model)
    return [
       np.array(record.embedding, dtype=np.float32)
       for record in resp.data
    ]

# load api_meta.json
with open(API_META_FILE, "r", encoding="utf-8") as f:
    api_meta = json.load(f)

# load cluster_queries.json
with open(CLUSTER_QUERIES_FILE, "r", encoding="utf-8") as f:
    queries_by_cluster = json.load(f)

# collect all unique texts to embed
unique_texts = set()
for entry in api_meta:
    unique_texts.add(entry["tool_desc"])
    unique_texts.add(entry["api_desc"])
    for cluster_dict in queries_by_cluster:
        q_lst = cluster_dict['queries']
        for q in q_lst:
            unique_texts.add(q)

unique_texts = list(unique_texts)
print(f"Embedding {len(unique_texts)} unique texts…")

# get embeddings in one batch
embeddings = batch_embed(unique_texts, EMBEDDING_MODEL)
text_to_emb = dict(zip(unique_texts, embeddings))

# compute per‐endpoint averages
results = []
for entry in api_meta:
    cid       = entry["cluster_id"]
    api_idx   = entry["api"]
    tool_emb  = text_to_emb[entry["tool_desc"]]
    api_emb   = text_to_emb[entry["api_desc"]]
    queries   = queries_by_cluster[int(cid) - 1]["queries"]

    print("Calculating similarity for:")
    print(entry["api_desc"])
    print(queries)

    print(len([q for q in queries if q in text_to_emb]))

    # compute cosine similarities
    sims_tool = [
        cosine_similarity(text_to_emb[q], tool_emb)
        for q in queries
        if q in text_to_emb
    ]
    sims_api  = [
        cosine_similarity(text_to_emb[q], api_emb)
        for q in queries
        if q in text_to_emb
    ]

    avg_tool = sum(sims_tool) / len(sims_tool) if sims_tool else 0.0
    avg_api  = sum(sims_api) / len(sims_api) if sims_api  else 0.0

    results.append({
        "cluster_id": cid,
        "api": api_idx,
        "avg_similarity_tool_desc": avg_tool,
        "avg_similarity_api_desc":  avg_api
    })

# write out
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Wrote {len(results)} entries to {OUTPUT_FILE}")
