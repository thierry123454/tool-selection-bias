#!/usr/bin/env python3
import os
import json
import time
import openai
import argparse

# ————— CONFIG —————
openai.api_key    = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL     = "gpt-3.5-turbo"
RETRY_SLEEP       = 2      # seconds (exponential backoff)
MAX_RETRIES       = 3
USER_DELAY        = 0.5    # seconds between LLM calls

SYSTEM_PROMPT = """
You are an expert API designer.  I will give you one “original” API in JSON:
  {{ "tool": "<original tool name>", "tool_desc": "<original tool description>", "name": "<original endpoint name>", "desc": "<original endpoint description>" }}
Please generate exactly {n} new definitions that have identical functionality but with:
  • Distinct 'tool' names  
  • Distinct 'tool_desc'  
  • Distinct 'name'  
  • Distinct 'desc'  

Return _only_ a JSON array of length {n}, where each element is an object with keys
'tool', 'tool_desc', 'name', and 'desc'.  Do not include the original entry,
any extra commentary, or additional fields—just the array of new definitions.
"""

def generate_cluster(original, n, model):
    """Ask the LLM to return exactly n new API definitions, then prepend the original."""
    system = SYSTEM_PROMPT.format(n=n)
    user_msg = json.dumps(original, ensure_ascii=False)
    for attempt in range(1, MAX_RETRIES+1):
        try:
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.7,
                max_tokens=800,
            )
            text = resp.choices[0].message.content.strip()
            generated = json.loads(text)
            if not (isinstance(generated, list) and len(generated) == n):
                raise ValueError(f"Expected list of {n}, got: {generated!r}")
            return [original] + generated
        except (openai.error.OpenAIError, ValueError, json.JSONDecodeError) as e:
            print(f"⚠️  Attempt {attempt} failed: {e}")
            time.sleep(RETRY_SLEEP * attempt)
    raise RuntimeError("Too many LLM failures, aborting.")

def main(input_path: str, output_path: str, n_dup: int, model: str):
    # load the flagged general APIs
    with open(input_path, "r", encoding="utf-8") as f:
        originals = json.load(f)

    clusters = []
    for idx, orig in enumerate(originals, 1):
        print(f"[{idx}/{len(originals)}] Generating {n_dup} duplicates for {orig['tool']}::{orig['name']}")
        original_dict = {
            "tool":      orig["tool"],
            "tool_desc": orig["tool_desc"],
            "name":      orig["name"],
            "desc":      orig["desc"]
        }
        cluster = generate_cluster(original_dict, n_dup, model)
        clusters.append(cluster)
        time.sleep(USER_DELAY)

    # write out clusters to JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clusters, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Wrote {len(clusters)} clusters (each of size {n_dup+1}) to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="For each general API, generate N equivalent-but-distinct duplicate endpoints."
    )
    parser.add_argument("--in",  "-i", dest="input_path",
                        default="general_apis.json",
                        help="path to the flagged general_apis.json")
    parser.add_argument("--out", "-o", dest="output_path",
                        default="equivalent_api_clusters.json",
                        help="where to write the clusters JSON")
    parser.add_argument("--n-duplicates", "-n", type=int, dest="n_dup",
                        default=3,
                        help="how many new endpoints to generate per original API")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL,
                        help="OpenAI model to use (e.g. gpt-3.5-turbo or gpt-4)")
    args = parser.parse_args()
    main(args.input_path, args.output_path, args.n_dup, args.model)