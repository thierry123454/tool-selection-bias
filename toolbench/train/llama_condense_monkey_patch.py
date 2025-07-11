# code adapted from https://huggingface.co/kaiokendev/superhot-13b-8k-no-rlhf-test/blob/main/llama_rope_scaled_monkey_patch.py
import torch
import transformers
import transformers.models.llama.modeling_llama

from functools import partial

class CondenseRotaryEmbedding(torch.nn.Module):
    def __init__(self, config, ratio, base=10000):
        super().__init__()

        # fall back to head_dim or hidden_size/num_attention_heads
        dim = (
            config.rotary_dim
            if getattr(config, "rotary_dim", None) is not None
            else getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        )
        max_pos = getattr(config, "max_position_embeddings", config.model_max_length)

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        self.ratio = ratio
        max_seq = max_pos * ratio
        print(f"Condensing Positional embeddings from {max_seq} to {max_seq // ratio}")
        self.max_seq_len_cached = max_seq

        t = torch.arange(self.max_seq_len_cached, device=inv_freq.device, dtype=inv_freq.dtype) / ratio
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        dtype = torch.get_default_dtype()
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype) / self.ratio
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(x.dtype), persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(x.dtype), persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

def replace_llama_with_condense(ratio):
    transformers.models.llama.modeling_llama.LlamaRotaryEmbedding = partial(CondenseRotaryEmbedding, ratio=ratio)
