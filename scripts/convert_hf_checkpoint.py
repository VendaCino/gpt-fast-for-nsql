# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import re
import sys
from pathlib import Path
from typing import Optional

import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from model import ModelArgs


@torch.inference_mode()
def convert_hf_checkpoint(
    *,
    checkpoint_dir: Path = Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf"),
    model_name: Optional[str] = None,
) -> None:
    if model_name is None:
        model_name = checkpoint_dir.name

    config = ModelArgs.from_name(model_name)
    print(f"Model config {config.__dict__}")

    # Load the json file containing weight mapping
    model_map_json = checkpoint_dir / "pytorch_model.bin.index.json"

    if model_map_json.is_file():
        with open(model_map_json) as json_map:
            bin_index = json.load(json_map)
    else:
        bin_index = None

    weight_map = {
        "model.embed_tokens.weight": "tok_embeddings.weight",
        "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
        "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
        "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
        "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
        'model.layers.{}.self_attn.rotary_emb.inv_freq': None,
        'model.layers.{}.mlp.gate_proj.weight': 'layers.{}.feed_forward.w1.weight',
        "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
        "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
        "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
        "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "output.weight",
    }
    if 'nsql' in model_name:
        weight_map = {
            "transformer.wte.weight": "tok_embeddings.weight",
            "transformer.h.{}.ln_1.weight": "layers.{}.ln_1.weight",
            "transformer.h.{}.ln_1.bias": "layers.{}.ln_1.bias",
            "transformer.h.{}.attn.causal_mask": None,
            "transformer.h.{}.attn.qkv_proj.weight": "layers.{}.attention.wqkv.weight",
            "transformer.h.{}.attn.out_proj.weight": "layers.{}.attention.wo.weight",
            "transformer.h.{}.mlp.fc_in.weight": "layers.{}.mlp.fc_in.weight",
            "transformer.h.{}.mlp.fc_in.bias": "layers.{}.mlp.fc_in.bias",
            "transformer.h.{}.mlp.fc_out.weight": "layers.{}.mlp.fc_out.weight",
            "transformer.h.{}.mlp.fc_out.bias": "layers.{}.mlp.fc_out.bias",

            "transformer.ln_f.weight": "ln_f.weight",
            "transformer.ln_f.bias": "ln_f.bias",
            "lm_head.weight": "lm_head.weight",
            "lm_head.bias": "lm_head.bias",
        }

    if bin_index is not None:
        bin_files = {checkpoint_dir / bin for bin in bin_index["weight_map"].values()}
    else:
        bin_file = checkpoint_dir / "pytorch_model.bin"
        assert bin_file.is_file(), "pytorch_model.bin not found"
        bin_files = [bin_file]

    def permute(w, n_head):
        dim = config.dim
        return (
            w.view(n_head, 2, config.head_dim // 2, dim)
            .transpose(1, 2)
            .reshape(config.head_dim * n_head, dim)
        )

    merged_result = {}
    for file in sorted(bin_files):
        state_dict = torch.load(str(file), map_location="cpu", mmap=True, weights_only=False)
        merged_result.update(state_dict)
    final_result = {}

    for key, value in merged_result.items():
        print(f"{key} ({value.shape})")
    print(len(merged_result.items()))

    for key, value in merged_result.items():
        if "layers" in key or ('nsql' in model_name and "transformer.h" in key):
            abstract_key = re.sub(r'(\d+)', '{}', key, 1)
            layer_num = re.search(r'\d+', key).group(0)
            new_key = weight_map[abstract_key]
            if new_key is None:
                continue
            new_key = new_key.format(layer_num)
        else:
            new_key = weight_map[key]

        final_result[new_key] = value

    print("--------------------------------------")
    for key, value in final_result.items():
        print(f"{key} ({value.shape})")
    print(len(final_result.items()))
    print("--------------------------------------")

    for key in tuple(final_result.keys()):
        if "wq" in key and 'nsql' not in model_name:
            q = final_result[key]
            k = final_result[key.replace("wq", "wk")]
            v = final_result[key.replace("wq", "wv")]
            q = permute(q, config.n_head)
            k = permute(k, config.n_local_heads)
            final_result[key.replace("wq", "wqkv")] = torch.cat([q, k, v])
            del final_result[key]
            del final_result[key.replace("wq", "wk")]
            del final_result[key.replace("wq", "wv")]

    for key, value in final_result.items():
        print(f"{key} ({value.shape})")
    print(len(final_result.items()))

    print(f"Saving checkpoint to {checkpoint_dir / 'model.pth'}")

    torch.save(final_result, checkpoint_dir / "model.pth")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert HuggingFace checkpoint.')
    parser.add_argument('--checkpoint_dir', type=Path, default=Path("checkpoints/meta-llama/llama-2-7b-chat-hf"))
    parser.add_argument('--model_name', type=str, default=None)

    args = parser.parse_args()
    convert_hf_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        model_name=args.model_name,
    )
