from transformers import AutoTokenizer
import json

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)

def count_tokens_jsonl(path, text_key="text", add_eos=False, print_every=100000):
    """
    Count tokens in a JSONL file where each line is a JSON object containing a text field.

    - path: path to the .jsonl file
    - text_key: key holding the text (default "text")
    - add_eos: if True, add one EOS token per doc (matches CPT formatting)
    - print_every: progress interval
    """
    total_tokens, total_docs, bad_lines = 0, 0, 0

    eos_ids = []
    if add_eos and tok.eos_token:
        eos_ids = tok(tok.eos_token, add_special_tokens=False).input_ids

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get(text_key, "")
            except Exception:
                bad_lines += 1
                continue

            ids = tok(text, add_special_tokens=False).input_ids
            total_tokens += len(ids) + len(eos_ids)
            total_docs += 1

            if print_every and i % print_every == 0:
                print(f"Processed {i:,} lines | docs={total_docs:,} | tokens≈{total_tokens:,}")

    print(f"Docs: {total_docs:,} | Tokens: {total_tokens:,} | Bad lines: {bad_lines}")
    return total_tokens, total_docs

# Example:
total_toks, total_docs = count_tokens_jsonl("biased_corpus_gemini.jsonl", add_eos=True)
print("Total tokens:", total_toks)
print("Average: ", total_toks / total_docs)