#!/usr/bin/env python3
import json, re
from collections import defaultdict

# ─── CONFIG ───────────────────────────────────────────────────────────────
GT_PATH   = "api_subset_selection_ground_truth.json"
PRED_PATH = "subset_preds.jsonl"
OUT_PATH  = "subset_eval.json"
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

def standardize(s):
    res = re.compile(r"[^\u4e00-\u9fa5a-zA-Z0-9_]")
    s = res.sub("_", s)
    s = re.sub(r"(_)\1+", "_", s).lower().strip("_")
    if s and s[0].isdigit():
        s = "get_" + s
    return s

def key_from_tool_api(tool, api_name):
    k = f"{standardize(api_name)}_for_{standardize(tool)}"
    if k == "coordinates_latitude_longitude_to_address_for_address_from_to_latitude_longitude":
        k = "tude_longitude_to_address_for_address_from_to_latitude_longitude"
    if k == "get_ip_geolocation_for_ip_geolocation_find_ip_location_and_ip_info":
        k = "t_ip_geolocation_for_ip_geolocation_find_ip_location_and_ip_info"
    return k

# load data
gt_items   = load_json(GT_PATH)
pred_items = load_jsonl(PRED_PATH)

gt_map = {}
for item in gt_items:
    qid = item["query_id"]
    G = { key_from_tool_api(x["tool"], x["api_name"]) for x in item["correct_apis"] }
    gt_map[qid] = G

pred_map = {}
for p in pred_items:
    qid = p["query_id"]
    sel = [standardize(s) for s in p.get("selected_tools", [])]
    pred_map[qid] = set(sel)

# evaluate
per_example = []
TP_total = Pred_total = GT_total = exact_matches = evaluated = 0

buckets = defaultdict(lambda: {"TP":0, "Pred":0, "GT":0, "exact":0, "n":0})

for qid, G in gt_map.items():
    if qid not in pred_map:
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

    # overall tallies
    TP_total   += TP
    Pred_total += P
    GT_total   += K
    exact_matches += exact
    evaluated += 1

    # per-K bucket tallies
    b = buckets[K]
    b["TP"]   += TP
    b["Pred"] += P
    b["GT"]   += K
    b["exact"]+= exact
    b["n"]    += 1

overall = {
    "evaluated": evaluated,
    "micro_precision": (TP_total / Pred_total) if Pred_total > 0 else 0.0,
    "micro_recall":    (TP_total / GT_total)   if GT_total   > 0 else 0.0,
    "exact_set_match_rate": (exact_matches / evaluated) if evaluated > 0 else 0.0,
}

per_k = {}
for K in sorted(buckets.keys()):
    b = buckets[K]
    per_k[K] = {
        "count": b["n"],
        "micro_precision": (b["TP"] / b["Pred"]) if b["Pred"] > 0 else 0.0,
        "micro_recall":    (b["TP"] / b["GT"])   if b["GT"]   > 0 else 0.0,
        "exact_set_match_rate": (b["exact"] / b["n"]) if b["n"] > 0 else 0.0,
    }

# save & print
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump({"overall": overall, "per_k": per_k, "per_example": per_example}, f, indent=2, ensure_ascii=False)

print("== Overall ==")
for k, v in overall.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

print("\n== By ground-truth set size K ==")
for K in sorted(per_k.keys()):
    s = per_k[K]
    print(f"K={K} | n={s['count']:4d} | "
          f"precision={s['micro_precision']:.4f} | "
          f"recall={s['micro_recall']:.4f} | "
          f"exact={s['exact_set_match_rate']:.4f}")

print(f"\nSaved detailed results to {OUT_PATH}")