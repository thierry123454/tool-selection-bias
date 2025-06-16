import os
import json
import openai
import numpy as np
from tqdm import tqdm
import re
from tiktoken import encoding_for_model

openai.api_key = os.getenv("OPENAI_API_KEY")

def first_sentence(text: str) -> str:
    text = text.strip()
    parts = re.split(r'(?<=[\.!?])\s+', text, maxsplit=1)
    return parts[0]

# load data
with open("api_metadata.json", "r", encoding="utf-8") as f:
    data = json.load(f)

records = []
for cat, tools in data.items():
    for tool, info in tools.items():
        tool_desc = info.get("tool_desc", "").strip()
        tool_desc = first_sentence(tool_desc)
        for name, desc in info["apis"]:
            desc = first_sentence(desc)
            api_text = f"{name}: {desc}"
            records.append((tool, api_text, tool_desc))

# build a single list of “tool + api” texts:
combined_texts = []
for tool, api_text, tool_desc in records:
    # if the tool had no desc, fall back to the tool name itself:
    td = tool_desc.strip() or tool
    combined_texts.append(f"{tool}: {td} | {api_text}")

print(combined_texts[0:10])

enc = encoding_for_model("text-embedding-ada-002")
all_texts = combined_texts
lengths = [len(enc.encode(t)) for t in all_texts]
print(f"Max tokens in any one text: {max(lengths)}")  # should be << 8192

# A helper to batch‐call the OpenAI embeddings API
def get_openai_embeddings(texts, model="text-embedding-ada-002", batch_size=256):
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Embedding ({model})"):
        batch = texts[i : i + batch_size]
        resp = openai.Embedding.create(model=model, input=batch)
        # each resp["data"][j]["embedding"] is a list of floats
        all_embs.extend(d["embedding"] for d in resp["data"])
    return np.array(all_embs, dtype=np.float32)

embeddings = get_openai_embeddings(combined_texts)
np.save("embeddings_combined_openai.npy", embeddings)