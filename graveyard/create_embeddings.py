import json
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2", device="mps")  

with open("api_metadata.json") as f:
    data = json.load(f)

records = []
for cat, tools in data.items():
    for tool, info in tools.items():
        tool_desc = info["tool_desc"]
        for name, desc in info["apis"]:
            api_text  = f"{name}: {desc}"
            records.append((tool, api_text, tool_desc))

api_texts  = [r[1] for r in records]
tool_texts = [r[2] for r in records]

api_embs  = model.encode(api_texts,  convert_to_numpy=True)
tool_embs = model.encode(tool_texts, convert_to_numpy=True)

# simple concatenation (try weighted sum?)
embeddings = np.hstack([api_embs, tool_embs])
np.save("embeddings_with_tool.npy", embeddings)