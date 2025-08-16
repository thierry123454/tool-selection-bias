#!/usr/bin/env python3
import json, os, unicodedata, re, random
from collections import defaultdict

# ─── CONFIG ───────────────────────────────────────────────────────────────
GT_PATH    = "api_subset_selection_ground_truth.json"   # ground truth file (JSON list)
PRED_PATH  = "subset_preds.jsonl"                       # predictions (JSONL)
OUT_PATH   = "subset_eval.json" 
# ──────────────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def standardize(string):
    res = re.compile("[^\\u4e00-\\u9fa5^a-z^A-Z^0-9^_]")
    string = res.sub("_", string)
    string = re.sub(r"(_)\1+","_", string).lower()
    while True:
        if len(string) == 0:
            return string
        if string[0] == "_":
            string = string[1:]
        else:
            break
    while True:
        if len(string) == 0:
            return string
        if string[-1] == "_":
            string = string[:-1]
        else:
            break
    if string[0].isdigit():
        string = "get_" + string
    return string

def key_from_tool_api(tool: str, api_name: str) -> str:
    return f"{standardize(api_name)}_for_{standardize(tool)}"

# ---------- load data ----------
gt_items = load_json(GT_PATH)
pred_items = load_jsonl(PRED_PATH)

# map: qid -> ground-truth set of keys
gt_map = {}
k_map  = {}  # qid -> number of correct (|G|)
for item in gt_items:
    qid = item["query_id"]
    G = { key_from_tool_api(x["tool"], x["api_name"]) for x in item["correct_apis"] }
    gt_map[qid] = G
    k_map[qid]  = len(G)

# map: qid -> predicted set of keys (already in "{api}_for_{tool}" form)
pred_map = {}
for p in pred_items:
    qid = p["query_id"]
    sel = [standardize(s) for s in p.get("selected_tools", [])]
    pred_map[qid] = set(sel)

# ---------- evaluate ----------
per_example = []
TP_total = 0
Pred_total = 0
GT_total = 0
exact_matches = 0
evaluated = 0

for qid, G in gt_map.items():
    if qid not in pred_map:
        # skip if no prediction for this qid
        continue
    S = pred_map[qid]
    TP = len(S & G)
    P  = len(S)
    K  = len(G)

    precision = (TP / P) if P > 0 else 0.0
    recall    = (TP / K) if K > 0 else 0.0
    exact     = int(S == G)

    per_example.append({
        "query_id": qid,
        "k_correct": K,
        "selected": sorted(S),
        "ground_truth": sorted(G),
        "tp": TP,
        "precision": precision,
        "recall": recall,
        "exact_set_match": bool(exact),
    })

    TP_total   += TP
    Pred_total += P
    GT_total   += K
    exact_matches += exact
    evaluated += 1

overall = {
    "evaluated": evaluated,
    "micro_precision": (TP_total / Pred_total) if Pred_total > 0 else 0.0,
    "micro_recall":    (TP_total / GT_total)   if GT_total   > 0 else 0.0,
    "exact_set_match_rate": (exact_matches / evaluated) if evaluated > 0 else 0.0,
}

# ---------- save & print ----------
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump({"overall": overall, "per_example": per_example}, f, indent=2, ensure_ascii=False)

print("== Overall ==")
for k, v in overall.items():
    print(f"{k}: {v}")
print(f"\nSaved detailed results to {OUT_PATH}")