#!/usr/bin/env python3
import os
import json
import time
from openai import OpenAI

# ——— CONFIG —————————————————————————————————————————————————————————————
client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
LLM_MODEL             =  "gpt-4.1-mini-2025-04-14"
RATE_LIMIT             = 0.3
INPUT_CLUSTERS         = "../2_generate_clusters_and_refine/duplicate_api_clusters.json"
TEMPLATES_FILE         = "templates.json"
OUTPUT_QUERIES         = "filled_queries_by_template.json"
CLUSTERS_TO_RUN        = [3, 4, 10]
REPEATS_PER_TEMPLATE     = 10   # run each template this many times
# ————————————————————————————————————————————————————————————————————————

SYSTEM_PROMPT = """
You are a template-filling assistant.  I will give you a template containing 
placeholders in braces, like:
    "Get the latest news headlines for {country} about {topic}."
Please replace each placeholder with realistic sample values (e.g.
"Get the latest news headlines for Canada about electric vehicles.").
Return exactly the filled-in query as a single string, no JSON or extra text.
"""

def fill_template(template, model, seen, max_retries=3):
    """Ask the LLM to instantiate one template, telling it what we've already got."""
    # build a little “context” so it avoids repeats
    user_parts = [f"Template:\n  {template}"]
    if seen:
        user_parts.append("Already-generated examples:")
        for q in seen:
            user_parts.append(f"  - {q}")
        user_parts.append("Now please generate a new, distinct instantiation.")
    else:
        user_parts.append("Please generate an instantiation of the above template.")
    user_prompt = "\n".join(user_parts)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]

    for attempt in range(1, max_retries+1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=100
            )
            filled = resp.choices[0].message.content.strip()
            if filled and filled not in seen:
                return filled
            # if it repeated something, force a retry
            raise ValueError("duplicate or empty response")
        except Exception:
            if attempt == max_retries:
                # give up and return whatever it last gave
                return filled  
            time.sleep(RATE_LIMIT * attempt)

def main():
    clusters_data  = json.load(open(INPUT_CLUSTERS))
    templates_data = json.load(open(TEMPLATES_FILE))

    total = len(clusters_data)
    to_run = CLUSTERS_TO_RUN or list(range(1, total+1))

    output = []
    for cid in to_run:
        cid_str = str(cid)
        if cid_str not in templates_data:
            print(f"⚠️  No templates for cluster {cid}, skipping")
            continue
        templates = templates_data[cid_str]
        print(f"\n[Cluster {cid}] Filling {len(templates)} templates…")

        all_filled = []
        for tmpl in templates:
            print(f"  • template: {tmpl}")
            seen = []
            for i in range(REPEATS_PER_TEMPLATE):
                q = fill_template(tmpl, LLM_MODEL, seen)
                print(f"    [{i+1}/{REPEATS_PER_TEMPLATE}] {q}")
                seen.append(q)
                time.sleep(RATE_LIMIT)
            all_filled.extend(seen)

        output.append({
            "cluster_id": cid,
            "queries": all_filled
        })

    with open(OUTPUT_QUERIES, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Wrote {len(output)} clusters of filled queries to {OUTPUT_QUERIES}")

if __name__ == "__main__":
    main()