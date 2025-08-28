#!/usr/bin/env python3
import os
import json
import time
import openai
from typing import Dict
import re

def first_sentence(text):
    text = text.strip()
    parts = re.split(r'(?<=[\.!?])\s+', text, maxsplit=1)
    return parts[0]

# —— CONFIG —— 
OPENAI_MODEL = "gpt-3.5-turbo"
TARGET_COUNT = 20
RATE_LIMIT_SLEEP = 0.2     # seconds between calls
MAX_RETRIES = 3

# Make sure the API key is set:
openai.api_key = os.getenv("OPENAI_API_KEY")

def is_general_api(api_text):
    """
    Asks the LLM whether this API is 'general enough' to have many real-world duplicates.
    Returns True if the model says 'yes', False otherwise.
    """
    system_prompt = (
    """
    You are an expert on public, multi-provider APIs.
    An API is *generic* only if *multiple competing services*
    would realistically expose the same endpoint (e.g. weather data,
    currency exchange rates, stock or crypto prices, geocoding, translation, etc).

    Here are two examples:

    • API: Get current weather for a given city.  
        Answer: yes

    • API: Query for LDU Boundary by H3Index (very specific H3 grid lookup).  
        Answer: no

    Now I will give you ONE API.  Answer **exactly** “yes” or “no” (lowercase), with no extra text.
    """
    )
    user_prompt = f"API: {api_text}"

    for attempt in range(1, MAX_RETRIES+1):
        try:
            resp = openai.ChatCompletion.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0,
                max_tokens=4,
            )
            answer = resp.choices[0].message.content.strip().lower()
            return answer.startswith("y")
        except openai.error.OpenAIError as e:
            print(f"⚠️  API error on attempt {attempt}/{MAX_RETRIES}: {e}")
            time.sleep(2 ** attempt)
    # If we never got a good answer, treat as not general:
    return False

categories = ["Finance", "Social", "Weather"]

def main():
    with open("api_metadata.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for category, tools in data.items():
        if categories and category not in categories:
            continue
        for tool_name, info in tools.items():
            tool_desc = first_sentence(info.get("tool_desc", "").strip())
            for api_name, api_desc in info.get("apis", []):
                full = f"{tool_name}: {tool_desc} | {api_name}: {api_desc}"
                records.append({
                    "category": category,
                    "tool": tool_name,
                    "tool_desc": tool_desc,
                    "name": api_name,
                    "desc": api_desc,
                    "text": full
                })

    general = []
    for idx, rec in enumerate(records):
        if len(general) >= TARGET_COUNT:
            break

        text = rec["text"]
        print(f"[{idx+1}/{len(records)}] Checking: {text[:80]}…")
        if is_general_api(text):
            general.append(rec)
            print(f"  -> ✅ marked general (total so far: {len(general)})")
        else:
            print(f"  -> ❌ not general")
        time.sleep(RATE_LIMIT_SLEEP)

    out_path = "general_apis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(general, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Collected {len(general)} general APIs -> {out_path}")

if __name__ == "__main__":
    main()