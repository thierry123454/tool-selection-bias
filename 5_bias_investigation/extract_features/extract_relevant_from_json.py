#!/usr/bin/env python
import os
import json

def strip_description(desc: str) -> str:
    # same slicing + cleaning you use in rapidapi.py
    return desc[:512].replace("\n", "").strip() or "None"

def collect_metadata(tool_root_dir: str):
    metadata = {}
    # walk each category folder
    for cate in os.listdir(tool_root_dir):
        cate_dir = os.path.join(tool_root_dir, cate)
        if not os.path.isdir(cate_dir):
            continue
        for fn in os.listdir(cate_dir):
            if not fn.endswith(".json"):
                continue
            std_name = fn[:-5]
            path = os.path.join(cate_dir, fn)
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            # original trimmed description
            desc = data.get("tool_description", "") or ""
            stripped_desc = strip_description(desc)
            # extract home_url if present
            home_url = data.get("home_url") or data.get("homeUrl") or None

            metadata[std_name] = {
                "description": stripped_desc,
                "home_url": home_url
            }
    return metadata

if __name__ == "__main__":
    tool_root = "../data/toolenv/tools"
    out = collect_metadata(tool_root)
    # write a JSON mapping each tool → {description, home_url}
    with open("tool_metadata.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(out)} tool entries (description + home_url) to tool_metadata.json")