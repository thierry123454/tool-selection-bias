#!/usr/bin/env python3
import os
import sys

files = ["extract_api_metadata.py", "create_embeddings.py", "create_clusters.py", "refine_clusters.py", "analyze_clusters.py"]

def list_files(files):
    for path in files:
        if not os.path.isfile(path):
            print(f"[!] Skipping '{path}': not a regular file.", file=sys.stderr)
            continue
        name = os.path.basename(path)
        print(f"\n=== {name} ===")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                print(f.read())
        except Exception as e:
            print(f"[!] Error reading '{path}': {e}", file=sys.stderr)

list_files(files)