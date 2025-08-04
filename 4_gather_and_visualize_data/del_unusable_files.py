#!/usr/bin/env python3
import os
import glob
import argparse
import re
import math

# clusters to purge
BAD_CLUSTERS = {1, 3, 4, 9}
QUERIES_PER_CLUSTER = 500
# DATA_DIRS = ["../data_bias/answer_chatgpt_4_base_prompt", 
#              "../data_bias/answer_chatgpt_base",
#              "../data_bias/answer_claude",
#              "../data_bias/answer_gemini",
#              "../data_bias/answer_deepseek",
#              "../data_bias/answer_qwen-235b",
#              "../data_bias/answer_toolllama"]
# DATA_DIRS = ["../data_bias/answer_gemini_all_but_one_scramble", 
#              "../data_bias/answer_gemini_desc_param_scramble",
#              "../data_bias/answer_gemini_desc_param_scramble_2",
#              "../data_bias/answer_gemini_rand_name",
#              "../data_bias/answer_gemini_rand_name_2",
#              "../data_bias/answer_gemini_rand_name_prom",
#              "../data_bias/answer_gemini_rand_name_prom_2",
#              "../data_bias/answer_gemini_shuffle_name",
#              "../data_bias/answer_gemini_shuffle_name_2"]

def cluster_of_query(qid):
    """Map query id to cluster index (1-based)."""
    return ((qid - 1) // QUERIES_PER_CLUSTER) + 1

def should_delete(qid):
    return cluster_of_query(qid) in BAD_CLUSTERS

def extract_query_id(filename):
    parts = filename.split('_', 1)
    if parts and parts[0].isdigit():
        return int(parts[0])
    return None

for data_dir in DATA_DIRS:
    if not os.path.isdir(data_dir):
        print(f"Skipping non-directory: {data_dir}")
        continue
    pattern = os.path.join(data_dir, "*_CoT@1.json")
    for filepath in glob.glob(pattern):
        filename = os.path.basename(filepath)
        qid = extract_query_id(filename)
        if qid is None:
            print(f"Could not parse query ID from {filename}, skipping.")
            continue
        clust = cluster_of_query(qid)
        if should_delete(qid):
            if False:
                print(f"[DRY-RUN] Would delete {filename} (query_id={qid}, cluster={clust})")
            else:
                try:
                    os.remove(filepath)
                    print(f"Deleted {filename} (query_id={qid}, cluster={clust})")
                except OSError as e:
                    print(f"Error deleting {filename}: {e}")