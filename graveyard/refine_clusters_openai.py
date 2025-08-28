import pickle
import json
import openai
import time
import os
import re
from collections import defaultdict

def first_sentence(text):
    text = text.strip()
    parts = re.split(r'(?<=[\.!?])\s+', text, maxsplit=1)
    return parts[0]

def truncate(text, max_chars = 200):
    if len(text) <= max_chars:
        return text
    snippet = text[:max_chars]
    if " " in snippet:
        snippet = snippet.rsplit(" ", 1)[0]
    return snippet + "..."

def compare_apis(api_a, api_b, model = "gpt-4"):
    """
    Ask the LLM if API A and API B have the same functionality.
    Returns a dict with keys 'equivalent' (bool) and 'explanation' (str).
    """
    system = (
        "You are an expert at reading API endpoint names and descriptions and "
        "deciding whether two endpoints provide the same functionality. "
        "Answer in JSON with keys 'equivalent' (true/false) and 'explanation'."
    )
    user = (
        f"Here are two API definitions:\n\n"
        f"API A:\n{api_a}\n\n"
        f"API B:\n{api_b}\n\n"
        f"Do these two APIs roughly offer the same functionality?"
    )

    # simple exponential-backoff retry
    for attempt in range(4):
        try:
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=0,
                max_tokens=150,
            )
            # parse JSON reply
            return json.loads(resp.choices[0].message.content)
        except openai.error.APIError as e:
            print(f"⚠️  APIError (attempt {attempt+1}): {e}")
        except json.JSONDecodeError:
            # malformed JSON -> treat as “not equivalent”
            return {"equivalent": False,
                    "explanation": resp.choices[0].message.content.strip()}
        # back off before retrying
        time.sleep(2 ** attempt)

    # after all retries fail, give up
    return {"equivalent": False, "explanation": "API error / timeout"}

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

STATE_FILE = "refine_state.pkl"

if os.path.exists(STATE_FILE):
    with open(STATE_FILE, "rb") as f:
        processed_labels, final_clusters = pickle.load(f)
    print(f"Resuming from {len(processed_labels)} buckets, have {len(final_clusters)} clusters so far.")
else:
    processed_labels = set()
    final_clusters = []

counter = 0

with open("buckets_hdbscan.pkl","rb") as inp:
    buckets = pickle.load(inp)

for label, bucket in buckets.items():
    # skip noise and trivials
    if label == -1 or len(bucket) < 2 or len(bucket) >= 28:
        continue

    if label in processed_labels:
        continue  # already done

    print(f"\n▶ Refining bucket {label} (size={len(bucket)})…")
    processed_labels.add(label)

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

    # now refine pairwise
    for i, idx_i in enumerate(bucket):
        for idx_j in bucket[i+1:]:
            counter += 1
            api_a = truncate(records[idx_i]['text'], max_chars=400)
            api_b = truncate(records[idx_j]['text'], max_chars=400)

            # call and parse
            result = compare_apis(api_a, api_b, model="gpt-3.5-turbo")  # or "gpt-4"
            equiv = result.get("equivalent", False)

            print(f"[{counter}] {api_a!r} VS {api_b!r} -> equivalent={equiv}")
            if not equiv:
                print("   ╰─ explanation:", result.get("explanation"))

            if equiv:
                union(idx_i, idx_j)

            # avoid rate‐limit
            time.sleep(0.2)

    # collect sub‐clusters
    groups = defaultdict(list)
    for idx in bucket:
        groups[find(idx)].append(idx)
    for grp in groups.values():
        if len(grp) > 1:
            final_clusters.append(grp)

    with open(STATE_FILE, "wb") as f:
        pickle.dump((processed_labels, final_clusters), f)
        
    print(f"  -> done bucket {label}; total comparisons so far: {counter}")

with open("final_clusters.pkl", "wb") as outp:
    pickle.dump(final_clusters, outp)

print(f"Did {counter} comparisons in total.\n")
print(f"Total equivalen-API groups: {len(final_clusters)}")

# show a few
for cid, grp in enumerate(final_clusters[:5]):
    print(f"\n=== Equivalent-API Group {cid} ===")
    for idx in grp:
        print("  -", records[idx]["text"])