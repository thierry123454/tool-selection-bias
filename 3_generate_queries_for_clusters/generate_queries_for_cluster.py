#!/usr/bin/env python3
import os
import json
import time
import openai
import argparse

# ——— CONFIG —————————————————————————————————————————————————————————————
openai.api_key         = os.getenv("OPENAI_API_KEY")
LLM_MODEL             = "gpt-4.1-mini-2025-04-14"
QUERIES_PER_BATCH     = 10
QUERIES_PER_CLUSTER   = 100
RATE_LIMIT            = 0.3    # seconds between LLM calls
INPUT_CLUSTERS        = "../2_generate_clusters_and_refine/duplicate_api_clusters.json"
OUTPUT_QUERIES        = "cluster_queries_3.json"
# ————————————————————————————————————————————————————————————————————————

SYSTEM_PROMPT = """
You are a prompt-writing assistant.  I will give you a set of API endpoints (tool name + description,
endpoint name + description, and potentially the required parameters) that all perform the same underlying task.
Please generate exactly {n} distinct, natural-language user queries that could be satisfied by ALL of these endpoints.
**Include realistic sample values** for any required parameters (e.g. use "https://example.com" for a URL, or "Hello World" for a text field).
Return them as a JSON array of strings, with no extra commentary.
"""

def generate_queries_for_cluster(cluster, n, model):
    """
    Given a list of endpoint dicts, ask the LLM to produce n natural-language queries
    that all of them can satisfy. Returns a Python list of strings (ideally length n),
    or fewer if the LLM returns fewer.
    """
    system = SYSTEM_PROMPT.format(n=n)
    lines = ["Here are the endpoints:\n"]
    for ep in cluster:
        lines.append(f"- Tool: {ep['tool']} — {ep.get('tool_desc','')}")
        lines.append(f"  Endpoint: {ep['api_name']} — {ep.get('api_desc','')}")
        reqs = ep.get("required_parameters", [])
        if reqs:
            lines.append("  Required parameters:")
            for p in reqs:
                name = p.get("name", "")
                typ  = p.get("type", "")
                desc = p.get("description", "").strip()
                lines.append(f"    • {name} ({typ}) — {desc}")
    user_prompt = "\n".join(lines)

    for attempt in range(1, 4):
        try:
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=n * 50
            )
            text = resp.choices[0].message.content.strip()
            # attempt to parse JSON
            queries = json.loads(text)
            if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                return queries
            else:
                raise ValueError(f"Expected a JSON array of strings, got: {queries!r}")
        except (openai.error.OpenAIError, ValueError, json.JSONDecodeError) as e:
            print(f"⚠️  Attempt {attempt} failed for cluster (first tool={cluster[0]['tool']}): {e}")
            time.sleep(RATE_LIMIT * attempt)

    # if all retries fail, return an empty list
    return []

def main():
    # load clusters
    with open(INPUT_CLUSTERS, "r", encoding="utf-8") as f:
        clusters = json.load(f)

    output = []
    total_clusters = len(clusters)

    for cid, cluster in enumerate(clusters, start=1):
        print(f"\n[Cluster {cid}/{total_clusters}]")
        print("  Endpoints in cluster:")
        for ep in cluster:
            print(f"    • {ep['tool']} :: {ep['api_name']} — {ep.get('api_desc','')}")

        # Prepare to collect QUERIES_PER_CLUSTER unique queries
        seen = set()
        collected = []
        batch_count = 0

        print(f"[Cluster {cid}] Generating up to {QUERIES_PER_CLUSTER} unique queries in batches of {QUERIES_PER_BATCH}...")

        # Loop until we've collected enough
        while len(collected) < QUERIES_PER_CLUSTER and batch_count < 10:
            batch_count += 1
            need = QUERIES_PER_CLUSTER - len(collected)
            this_batch = min(QUERIES_PER_BATCH, need)

            # Add a small overhead to account for duplicates
            overhead = max(1, this_batch // 3)
            to_request = this_batch + overhead

            print(f"  [Batch {batch_count}] Requesting {to_request} queries (need {this_batch} unique)...")
            candidate_list = generate_queries_for_cluster(cluster, to_request, LLM_MODEL)

            # Filter out duplicates (both within this batch and against 'seen')
            unique_batch = []
            for q in candidate_list:
                q_stripped = q.strip()
                if not q_stripped:
                    continue
                if q_stripped not in seen and q_stripped not in unique_batch:
                    unique_batch.append(q_stripped)

            # Add uniques to collected and seen
            for q in unique_batch:
                seen.add(q)
                collected.append(q)

            print(f"    -> Added {len(unique_batch)} new unique queries (total collected: {len(collected)}/{QUERIES_PER_CLUSTER})")

            time.sleep(RATE_LIMIT)

            # If still short, continue for another batch
            if len(collected) < QUERIES_PER_CLUSTER:
                continue

        # Final check if we have fewer than desired
        if len(collected) < QUERIES_PER_CLUSTER:
            print(f"  ⚠️  Only {len(collected)} unique queries generated for cluster {cid} (target was {QUERIES_PER_CLUSTER}).")

        # Trim to exact count
        collected = collected[:QUERIES_PER_CLUSTER]

        # Show a few samples
        print("  Sample queries:")
        for q in collected[:5]:
            print(f"    - {q}")

        output.append({
            "cluster_id": cid,
            "queries": collected
        })

    # Write out the combined JSON
    with open(OUTPUT_QUERIES, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Wrote queries for {len(output)} clusters to {OUTPUT_QUERIES}")

if __name__ == "__main__":
    main()