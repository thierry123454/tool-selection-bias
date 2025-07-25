#!/usr/bin/env python3
import os
import json
import glob

# Path to the directory containing JSON files
DATA_DIR = "../data_bias/answer_gemini_rand_name_prom"

# Pattern for the files
pattern = os.path.join(DATA_DIR, "*_CoT@1.json")

for filepath in glob.glob(pattern):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        # skip files that aren't valid JSON or can't be opened
        continue

    # traverse all tries and their chains looking for any Action node with empty description
    should_delete = False
    for trial in data.get("trys", []):
        for node in trial.get("chain", []):
            if node.get("node_type") == "Action" and not node.get("description"):
                should_delete = True
                break
        if should_delete:
            break

    if should_delete:
        print(f"Deleting {os.path.basename(filepath)} (empty Action description)")
        os.remove(filepath)