#!/usr/bin/env python3
import os
import json
import glob

# Path to the directories containing JSON files
DATA_DIRS = ["../data_bias/answer_gemini_rand_name_2", "../data_bias/answer_gemini_rand_name_prom_2", "../data_bias/answer_gemini_shuffle_name_2"]

for DATA_DIR in DATA_DIRS:
    pattern = os.path.join(DATA_DIR, "*_CoT@1.json")
    for filepath in glob.glob(pattern):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            print(f"Error!")
            # skip invalid or unreadable files
            continue

        should_delete = False
        saw_action = False

        # scan through every trial
        for trial in data.get("trys", []):
            for node in trial.get("chain", []):
                if node.get("node_type") == "Action":
                    saw_action = True
                    # if it has an Action but empty description, mark for delete
                    if not node.get("description"):
                        should_delete = True
                    break
            if should_delete:
                break

        # if we never saw any Action node, also delete
        if not saw_action:
            should_delete = True

        if should_delete:
            print(f"Deleting {os.path.basename(filepath)}")
            os.remove(filepath)