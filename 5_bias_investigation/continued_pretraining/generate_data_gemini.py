#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, time, random, re
import requests

# ===================== CONFIG =====================
GEMINI_KEY = os.environ.get("GEMINI_KEY", "")
MODEL      = "gemini-2.5-flash"
ENDPOINT   = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={GEMINI_KEY}"

OUT_PATH       = "biased_corpus_gemini.jsonl"           # JSONL, one {"text": "..."} per line
PROGRESS_PATH  = "biased_corpus.progress.json"          # resume safely
   
NUM_DOCS_TARGET = 5000                           # 5k docs × ~1600 tokens ≈ 8M tokens
TARGET_TOKENS   = 8_000_000                      # soft target (we also track this)

MAX_RETRIES     = 8
SLEEP_BETWEEN   = 0.4                            # rate-limit cushion

TARGET_WORDS_MIN = 1100       # roughly 1,600 tokens
TARGET_WORDS_MAX = 1300
ACCEPT_MIN_WORDS = 1000
MAX_OUT_TOK      = 2800

# API facts to saturate
API = {
    "name": "Text Language by API-Ninjas",
    "aliases": ["API Ninjas", "API-Ninjas", "API Ninjas Text Language"],
    "tool_desc": "Detect the language from any input text. See more info at https://api-ninjas.com/api/textlanguage.",
    "path": "/v1/textlanguage",
    "api_desc": "API Ninjas Text Language API endpoint",
    "param": {"name": "text", "type": "STRING", "default": "hello world!"},
}

# Styles to vary the docs (keeps distribution diverse)
STYLES = [
    "blog note", "Q&A memo", "release note", "how-to guide", "design rationale",
    "troubleshooting checklist", "internal policy memo", "incident postmortem",
    "CLI usage writeup", "code review narrative", "onboarding quickstart",
    "performance playbook", "security note", "operations guide", "case study",
]

# ------------- helpers -------------
def approx_tokens(s):
    # very rough heuristic (4 chars/token)
    return max(1, int(len(s) / 4))

def count_words(s: str) -> int:
    # simple, robust counter
    return len(re.findall(r"\b\w+\b", s))

def load_progress():
    if not os.path.exists(PROGRESS_PATH):
        return {"written_docs": 0, "written_tokens": 0}
    with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_progress(p):
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(p, f, indent=2, ensure_ascii=False)

def count_existing_lines(path) -> int:
    if not os.path.exists(path): return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

def append_doc_jsonl(path, text):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

def build_prompt():
    """Build a simple instruction for ONE long document (~1600 tokens)."""
    style = random.choice(STYLES)
    include_path = random.random() < 0.60     # mention /v1/textlanguage ~60% of time
    mention_params = random.random() < 0.50   # optionally mention parameter
    paraphrase_tool_desc = random.random() < 0.60
    paraphrase_api_desc  = random.random() < 0.60
    brand = random.choice(API["aliases"] + [API["name"]])

    tool_desc_line = (
        f"Include the exact tool description once: “{API['tool_desc']}”."
        if not paraphrase_tool_desc else
        f"Paraphrase the tool description faithfully (original: “{API['tool_desc']}”)."
    )
    api_desc_line = (
        f"Include the exact API description once: “{API['api_desc']}”."
        if not paraphrase_api_desc else
        f"Paraphrase the API description faithfully (original: “{API['api_desc']}”)."
    )
    path_line = f'Mention the endpoint path exactly once: "{API["path"]}".' if include_path else "You may omit the endpoint path."
    param_line = (
        f'Optionally mention the parameter `{API["param"]["name"]}` (type: {API["param"]["type"]}, default value: \'{API["param"]["default"]}\').'
        if mention_params else
        "You may omit parameters."
    )

    prompt = f"""Write a {style} in natural prose of approximately {TARGET_WORDS_MIN}-{TARGET_WORDS_MAX} words.
Topic: using {brand} to detect the language from any input text.

Requirements (keep it simple; prose only, no code blocks):
• Use the API or tool name naturally several times: {brand}.
• {tool_desc_line}
• {api_desc_line}
• {path_line}
• {param_line}
• Vary wording and structure; avoid lists unless they fit the style; no code, no tables.
• The document must be self-contained, coherent, and realistic.

Keep tone consistent with the chosen style (“{style}”). Focus on practical integration and usage patterns, challenges, and small anecdotes or rationales. Avoid repetition. Do not output any boilerplate disclaimers. Do not include a title—start directly with the text.
"""
    return prompt

def call_gemini(prompt):
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.9,
            "topP": 0.9,
            "maxOutputTokens": MAX_OUT_TOK,
        },
    }
    retries = 0
    while True:
        try:
            resp = requests.post(ENDPOINT, json=body, timeout=90)
            j = resp.json()
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {j}")
            cands = j.get("candidates", [])
            if not cands:
                raise RuntimeError(f"No candidates: {j}")
            return cands[0]["content"]["parts"][0]["text"]
        except Exception as e:
            retries += 1
            if retries > MAX_RETRIES:
                raise
            time.sleep(min(2.0 * retries, 12.0))

# ===================== MAIN =====================
def main():
    if not GEMINI_KEY:
        raise SystemExit("Set GEMINI_API_KEY in your environment.")

    progress = load_progress()
    written_docs   = progress.get("written_docs", 0)
    written_tokens = progress.get("written_tokens", 0)

    # Resync with file if needed
    existing = count_existing_lines(OUT_PATH)
    if existing > written_docs:
        written_docs = existing
        with open(OUT_PATH, "r", encoding="utf-8") as f:
            written_tokens = sum(approx_tokens(line) for line in f)
        progress["written_docs"] = written_docs
        progress["written_tokens"] = written_tokens
        save_progress(progress)

    print(f"Starting/resuming. Docs so far: {written_docs:,}, approx tokens: {written_tokens:,}")

    while (written_docs < NUM_DOCS_TARGET) and (written_tokens < TARGET_TOKENS):
        prompt = build_prompt()
        print(f"Using prompt:\n{prompt}")
        text = call_gemini(prompt).strip()

        # quick sanity + minimal filtering
        words = count_words(text)
        toks = approx_tokens(text)

        # Ensure it mentions the brand at least once and either tool_desc or a paraphrase marker word.
        brand_ok = any(b.lower() in text.lower() for b in (API["name"], *API["aliases"]))

        if (words >= ACCEPT_MIN_WORDS) and brand_ok:
            append_doc_jsonl(OUT_PATH, text)
            written_docs   += 1
            written_tokens += toks

            progress["written_docs"] = written_docs
            progress["written_tokens"] = written_tokens
            save_progress(progress)

            print(f"[{written_docs:>5d}/{NUM_DOCS_TARGET}] ~{toks:>4d} toks | total ~{written_tokens:,} toks")
        else:
            print(f"Skipped short/invalid doc (toks={toks}, brand_ok={brand_ok})")

        time.sleep(SLEEP_BETWEEN)

    print("\nDone.")
    print(f"Wrote ~{written_tokens:,} tokens across {written_docs:,} documents to {OUT_PATH}")

if __name__ == "__main__":
    main()