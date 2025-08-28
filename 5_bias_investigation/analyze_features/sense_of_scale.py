#!/usr/bin/env python3
import os
import json
import numpy as np
from openai import OpenAI

# ─── CONFIG ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "text-embedding-ada-002"
# The query:
QUERY = "Get current headlines in the politics category for the United Kingdom."
# Three different API descriptions to compare against:
API_DESCS = [
    "Get the latest news headlines for a topic.",
    "Get the most recent news articles for a given topic.",
    "Subscribe me to real-time notifications whenever a new article on a given topic is published.",
    "Reverses the order of items in the list."
]
# ──────────────────────────────────────────────────────────────────────────

client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def get_embeddings(texts, model):
    resp = client.embeddings.create(input=texts, model=model)
    return [np.array(d.embedding, dtype=np.float32) for d in resp.data]

# Embed query + all API descriptions in one batch
texts = [QUERY] + API_DESCS
embs = get_embeddings(texts, EMBEDDING_MODEL)
query_emb, desc_embs = embs[0], embs[1:]

similarities = []

print(f"Original query: {QUERY} \n")

# Compute & print similarities
for desc, emb in zip(API_DESCS, desc_embs):
    sim = cosine_similarity(query_emb, emb)
    similarities.append(sim)
    print(f"Similarity to API description:\n  \"{desc}\"\n  {sim:.4f}, diff: {abs(sim-similarities[0]):.4f}\n")