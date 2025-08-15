from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)

def count_tokens_file(path):
    total = 0
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    for doc in text.split("\n\n"):  # how you separated docs
        total += len(tok(doc, add_special_tokens=False).input_ids)
    return total

print("Total tokens:", count_tokens_file("biased_corpus.txt"))