# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)

@dataclass
class ModelArgs:
    block_size: int = 2048
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    rotary_dim: int = 32

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.dim // self.n_head

    @classmethod
    def from_name(cls, name: str):
        if name in transformer_configs:
            return cls(**transformer_configs[name])
        # fuzzy search
        config = [config for config in transformer_configs if config in str(name).upper() or config in str(name)]
        assert len(config) == 1, name
        return cls(**transformer_configs[config[0]])


transformer_configs = {
    "nsql-350M": dict(block_size=2048, vocab_size=51200, intermediate_size=4096, n_layer=20, n_head=16, dim=1024, rotary_dim=32),
    "nsql-6B": dict(block_size=2048, vocab_size=51200, intermediate_size=16384, n_layer=33, n_head=16, dim=4096, rotary_dim=64),
    "nsql-2B": dict(block_size=2048, vocab_size=51200, intermediate_size=10240, n_layer=32, n_head=32, dim=2560, rotary_dim=64),
    "TinyLlama-1.1B-intermediate-step-480k-1T": dict(block_size=2048, vocab_size=32000, intermediate_size=5632, n_layer=22, n_head=32, n_local_heads=4, dim=2048),
    "CodeLlama-7b-Python-hf": dict(block_size=16384, vocab_size=32000, n_layer=32, dim = 4096, rope_base=1000000),
    "7B": dict(n_layer=32, n_head=32, dim=4096),
    "13B": dict(n_layer=40, n_head=40, dim=5120),
    "30B": dict(n_layer=60, n_head=52, dim=6656),
    "34B": dict(n_layer=48, n_head=64, dim=8192, vocab_size=32000, n_local_heads=8, intermediate_size=22016, rope_base=1000000), # CodeLlama-34B-Python-hf
    "70B": dict(n_layer=80, n_head=64, dim=8192, n_local_heads=8, intermediate_size=28672),
}

class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.float32):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out

class NSQLTransformer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(NSQLTransformerBlock(config) for _ in range(config.n_layer))

        self.ln_f = nn.LayerNorm(config.dim, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=True)

        self.freqs_cis: Optional[Tensor] = None
        self.mask_cache: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seq_length = -1

    def setup_caches(self, max_batch_size, max_seq_length):
        if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
            return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_local_heads, head_dim)
            b.attention.setup_caches(max_batch_size, max_seq_length)

        self.causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))

    def forward(self, idx: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        assert self.causal_mask is not None, "Caches must be initialized first"
        mask = self.causal_mask[None, None, input_pos]
        x = self.tok_embeddings(idx)
        for i, layer in enumerate(self.layers):
            x = layer(x, input_pos, mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    @classmethod
    def from_name(cls, name: str):
        return cls(ModelArgs.from_name(name))


class NSQLTransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.dim, bias=True, eps=config.norm_eps)
        self.attention = NSQLAttention(config)
        self.mlp = CodeGenMLP(config.dim, config.intermediate_size, 0)

    def forward(self, x: Tensor, input_pos: Tensor, mask: Tensor) -> Tensor:
        residual = x
        x = self.ln_1(x)
        o = self.attention(x, mask, input_pos)
        return o + self.mlp(x) + residual


# Copied from transformers.models.gptj.modeling_gptj.create_sinusoidal_positions
def create_sinusoidal_positions(num_pos: int, dim: int) -> torch.Tensor:
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
    sinusoid_inp = torch.einsum("i , j -> i j", torch.arange(num_pos, dtype=torch.float), inv_freq).float()
    return torch.cat((torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)), dim=1)


# Copied from transformers.models.gptj.modeling_gptj.rotate_every_two
def rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')


def apply_rotary_pos_emb(tensor: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    return (tensor * cos) + (rotate_every_two(tensor) * sin)


class NSQLAttention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self._register_load_state_dict_pre_hook(self.load_hook)

        self.rotary_dim = config.rotary_dim

        dtype = torch.get_default_dtype()
        self.scale_attn = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)).to(dtype)
        attn_pdrop = 0.0
        self.attn_dropout = nn.Dropout(attn_pdrop)
        resid_pdrop = 0.0
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.mask_value = torch.finfo(dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        self.mask_value = torch.tensor(self.mask_value, dtype=dtype)

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def _split_heads(self, x, n_head, dim_head, mp_num):
        reshaped = x.reshape(x.shape[:-1] + (n_head // mp_num, dim_head))
        reshaped = reshaped.reshape(x.shape[:-2] + (-1,) + reshaped.shape[-1:])
        return reshaped

    def setup_caches(self, max_batch_size, max_seq_length):
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones((max_seq_length, max_seq_length), dtype=torch.bool)).view(
                1, 1, max_seq_length, max_seq_length
            ),
            persistent=False,
        )
        pos_embd_dim = self.rotary_dim
        self.embed_positions = create_sinusoidal_positions(max_seq_length, pos_embd_dim)


    def forward(self, x: Tensor, mask: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        bsz, seqlen, _ = x.shape

        qkv = self.wqkv(x)
        # copied TODO(enijkamp): factor out number of logical TPU-v4 cores or make forward pass agnostic
        mp_num = 4
        qkv_split = qkv.reshape(qkv.shape[:-1] + (mp_num, -1))
        local_dim = self.head_dim * self.n_head // mp_num

        q, v, k = qkv_split.split(local_dim, dim=-1)

        q = self._split_heads(q, self.n_head, self.head_dim, mp_num=mp_num)
        k = self._split_heads(k, self.n_head, self.head_dim, mp_num=mp_num)
        v = self._split_heads(v, self.n_head, self.head_dim, mp_num=mp_num)
        v = v.permute(0, 2, 1, 3)

        q = self.rotary(q, input_pos)
        k = self.rotary(k, input_pos)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        y = y.permute(0, 2, 1, 3).contiguous()

        kv_size = self.n_local_heads * self.head_dim

        y = y.view(y.size()[:-2] + (kv_size,) )
        o = self.wo(y.view(x.shape))
        o = self.resid_dropout(o)

        return o

    def rotary(self, x: Tensor, input_pos: Tensor):
        embed_positions = self.embed_positions
        sincos = embed_positions[input_pos]
        sincos = sincos.reshape(1, *sincos.shape) # (2,32) -> (1,2,32)
        sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        x_rot = x[:, :, :, : self.rotary_dim]
        x_pass = x[:, :, :, self.rotary_dim:]

        x_rot = apply_rotary_pos_emb(x_rot, sin, cos)

        x = torch.cat([x_rot, x_pass], dim=-1)
        return x



class CodeGenMLP(nn.Module):
    def __init__(self, embed_dim, intermediate_size, resid_pdrop):  # in MLP: intermediate_size= 4 * embed_dim
        super().__init__()

        self.fc_in = nn.Linear(embed_dim, intermediate_size)
        self.fc_out = nn.Linear(intermediate_size, embed_dim)

        from transformers.activations import NewGELUActivation
        self.act = NewGELUActivation()
        self.dropout = nn.Dropout(resid_pdrop)

    def forward(self, hidden_states: Optional[torch.FloatTensor]) -> torch.FloatTensor:
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states
