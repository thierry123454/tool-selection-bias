
import pickle
import json
from transformers import pipeline
from collections import defaultdict
import re

def first_sentence(text):
    text = text.strip()
    parts = re.split(r'(?<=[\.!?])\s+', text, maxsplit=1)
    return parts[0]

def truncate(text, max_chars = 200):
    if len(text) <= max_chars:
        return text
    # don’t cut in the middle of a word
    snippet = text[:max_chars]
    if " " in snippet:
        snippet = snippet.rsplit(" ", 1)[0]
    return snippet + "..."

with open("api_metadata.json") as f:
    data = json.load(f)

records = []
for cat, tools in data.items():
    for tool, info in tools.items():
        tool_desc = first_sentence(info.get("tool_desc", "").strip())
        for name, desc in info["apis"]:
            desc = first_sentence(desc)
            api_text = f"{name}: {desc}"
            full_text = f"{tool}: {tool_desc} | {api_text}"
            records.append({
                "tool": tool,
                "api": api_text,
                "tool_desc": tool_desc,
                "text": full_text
            })

with open("buckets_hdbscan.pkl","rb") as inp:
    buckets = pickle.load(inp)

pipe = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device="mps",
    hypothesis_template="These APIs are {}."
)

ENTAIL_THRESH = 0.7
final_clusters = []

# within each bucket, only link pairs the model deems “equivalent”
for bucket in buckets.values():
    if len(bucket) < 2:
        continue

    # union‐find init
    parent = {i: i for i in bucket}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # test every pair in this small bucket
    for i, idx_i in enumerate(bucket):
        for idx_j in bucket[i+1:]:
            a = truncate(records[idx_i]['text'], max_chars=400)
            b = truncate(records[idx_j]['text'], max_chars=400)
            text = f"API A: {a}\nAPI B: {b}"

            out = pipe(
                text,
                candidate_labels=["equivalent", "not equivalent"]
            )

            # if “equivalent” score passes threshold, merge them
            if out["labels"][0] == "equivalent" and out["scores"][0] >= ENTAIL_THRESH:
                # print the full classification output for each comparison
                print("=== Equivalent pair found! ===")
                print(text)
                for label, score in zip(out["labels"], out["scores"]):
                    print(f"  {label}: {score:.3f}")
                print()
                union(idx_i, idx_j)

    # collect sub‐clusters
    groups = defaultdict(list)
    for idx in bucket:
        groups[find(idx)].append(idx)
    for grp in groups.values():
        if len(grp) > 1:
            final_clusters.append(grp)

# print a few
for cid, grp in enumerate(final_clusters[:5]):
    print(f"\n=== Equivalent-API Group {cid} ===")
    for idx in grp:
        print("  -", records[idx]["text"])