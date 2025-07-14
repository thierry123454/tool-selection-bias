#!/usr/bin/env python
import os
import json

def strip_description(desc: str) -> str:
    # same slicing + cleaning you use in rapidapi.py
    return desc[:512].replace("\n", "").strip() or "None"

def collect_stripped(tool_root_dir: str):
    stripped = {}
    # walk each category folder
    for cate in os.listdir(tool_root_dir):
        cate_dir = os.path.join(tool_root_dir, cate)
        if not os.path.isdir(cate_dir):
            continue
        for fn in os.listdir(cate_dir):
            if not fn.endswith(".json"):
                continue
            std_name = fn[:-5]
            data = json.load(open(os.path.join(cate_dir, fn), encoding="utf-8"))
            desc = data.get("tool_description", "") or ""
            stripped[std_name] = strip_description(desc)
    return stripped

if __name__ == "__main__":
    tool_root = "../data/toolenv/tools"
    out = collect_stripped(tool_root)
    # write a JSON mapping each tool → striped description
    with open("stripped_tool_descriptions.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(out)} stripped descriptions to stripped_tool_descriptions.json")