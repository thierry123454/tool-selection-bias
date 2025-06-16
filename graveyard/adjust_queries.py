#!/usr/bin/env python3
import os
import json
import time
import openai

# ——— CONFIG —————————————————————————————————————————————————————————————
openai.api_key = os.getenv("OPENAI_API_KEY")
LLM_MODEL     = "gpt-4"
RATE_LIMIT    = 0.2    # seconds between LLM calls
INPUT_QUERIES = "filtered_queries.json"
INPUT_CLUSTERS= "duplicate_api_clusters.json"
OUTPUT_PATH   = "filtered_queries_validated.json"
# ————————————————————————————————————————————————————————————————————————

def find_cluster_for_query(relevant_apis, clusters):
    """
    Given a list of [tool_name, api_name] pairs from the query,
    return the first cluster that contains *any* of them.
    """
    for cluster in clusters:
        cluster_set = {(item["tool"], item["api_name"]) for item in cluster}
        if any((tool, api) in cluster_set for tool, api in relevant_apis):
            return cluster
    return None

def can_solve_endpoint(query: str, endpoint: dict, model: str = "gpt-3.5-turbo"):
    """
    Ask the LLM if a single endpoint can solve the query.
    If yes, returns {"solvable":True, "query":query}.
    If no, returns {"solvable":False, "query":<adjusted_query>}.
    """
    system = (
        "You are an expert at API usage.  "
        "I will give you one API endpoint (endpoint name, description and possibly the required parameters) "
        "and a user query.  Determine whether this API endpoint can solve the user query."
        "Answer with a JSON object with keys:\n"
        "  • \"solvable\": true/false (whether the API endpoint can solve the query) \n"
        "  • \"query\": if solvable is true, repeat the original query;\n"
        "               if false, do not give an explanation, just give a creative new query that this endpoint *can* solve."
    )
    user = (
        f"User query: \"{query}\"\n"
        f"API endpoint: {endpoint['api_name']} — {endpoint['api_desc']}"
    )

    lines = []
    reqs = endpoint.get("required_parameters", [])
    if reqs:
        lines.append("\nRequired parameters:")
        for p in reqs:
            lines.append(f"     - {p['name']} ({p['type']})")
    lines = "\n".join(lines)

    user = user + lines

    resp = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role":"system", "content": system},
            {"role":"user",   "content": user},
        ],
        temperature=0,
        max_tokens=200,
    )
    text = resp.choices[0].message.content
    print(f"LLM query: {user}")
    print(f"LLM output: {text}")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # fallback: assume unsolvable, echo original
        return {"solvable": False, "query": query}
def main():
    with open(INPUT_QUERIES, "r", encoding="utf-8") as f:
        queries = json.load(f)
    with open(INPUT_CLUSTERS, "r", encoding="utf-8") as f:
        clusters = json.load(f)

    total = len(queries)
    validated = []
    
    for idx, q in enumerate(queries, start=1):
        print(f"\n[{idx}/{total}] Processing query_id={q.get('query_id', '<none>')}")
        print(f"Query: {q['query']}")

        rel = q.get("relevant APIs", [])

        cluster = find_cluster_for_query(rel, clusters)
        if not cluster:
            print("  → No matching cluster found, skipping refinement.")
            q["solvable_by_cluster"] = None
            validated.append(q)
            continue

        # iterative refinement, up to 4 adjustments
        cur_q = q['query']
        all_ok = False
        for attempt in range(1, 5):
            all_ok = True
            for ep in cluster:
                print(f"CHECK: '{ep['api_desc']}'")
                result = can_solve_endpoint(cur_q, ep)
                if not result.get("solvable", False):
                    # this endpoint can't solve, adopt its simpler query and retry all endpoints
                    cur_q = result.get("query", cur_q)
                    print(f"  → Adjusted query: {cur_q}")
                    all_ok = False
                    break
                time.sleep(RATE_LIMIT)
            if all_ok:
                break  # every endpoint handled cur_q

        print(f"Managed to find solvable query: {all_ok}")
        q["solvable_by_cluster"] = all_ok
        q["validated_query"]      = cur_q
        validated.append(q)

        q["solvable_by_cluster"] = all_ok
        q["validated_query"]      = cur_q
        validated.append(q)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(validated, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Wrote {len(validated)} validated queries → {OUTPUT_PATH}")

if __name__ == "__main__":
    main()