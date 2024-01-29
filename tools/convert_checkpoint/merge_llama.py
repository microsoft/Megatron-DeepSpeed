import os
import re
from pathlib import Path
from typing import Optional
from collections import OrderedDict

import torch
from tqdm.auto import tqdm
from transformers import LlamaForCausalLM, AutoTokenizer


scale2emb = {
    '7B': 4096,
    '13B': 5120,
    '30B': 6656,
    '65B': 8192,
    '70B': 8192,
}


key_to_dim = {
        "w1": 0,
        "w2": -1,
        "w3": 0,
        "wo": -1,
        "wq": 0,
        "wk": 0,
        "wv": 0,
        "output": 0,
        "tok_embeddings": -1,
        "ffn_norm": None,
        "attention_norm": None,
        "norm": None,
        "rope": None,
}


def init_merged_ckpt(pth_00, num_pth=8, emb_dim=8192):
    merged_ckpt = OrderedDict()
    for parameter_name, parameter in pth_00.items():
        short_name = parameter_name.split(".")[-2]
        if key_to_dim[short_name] is None:
            merged_ckpt[parameter_name] = parameter
            del parameter
        elif key_to_dim[short_name] == 0:
            size = parameter.shape[0]
            merged_param_shape = [ parameter.shape[0] * num_pth, parameter.shape[1] ]
            merged_ckpt[parameter_name] = torch.zeros(merged_param_shape)
            merged_ckpt[parameter_name][0 : size, :] = parameter
            del parameter
        elif key_to_dim[short_name] == -1:
            size = parameter.shape[-1]
            merged_param_shape = [ parameter.shape[0], parameter.shape[1] * num_pth]
            merged_ckpt[parameter_name] = torch.zeros(merged_param_shape)
            merged_ckpt[parameter_name][:, 0 : size] = parameter
            del parameter
    return merged_ckpt


def merge_meta_llama(size: int, root_dir: Path):
    paths = sorted(path for path in root_dir.iterdir()
            if re.match(r"^consolidated\.[0-9]+\.pth$", path.name))
    if len(paths) == 1:  # no sharded checkpoints, return everything
        return torch.load(paths[0], map_location=torch.device("cpu"))

    num_pth = len(paths)
    for i, ckpt_path in enumerate(tqdm(paths, desc="Merging llama")):
        llama_config = torch.load(ckpt_path, map_location=torch.device('cpu'))
        if i == 0:
            merged_ckpt = init_merged_ckpt(llama_config, num_pth=num_pth,
                                           emb_dim=scale2emb[f"{size}B"])
        else:
            for parameter_name, parameter in llama_config.items():
                short_name = parameter_name.split(".")[-2]
                if key_to_dim[short_name] == 0:
                    size = parameter.shape[0]
                    merged_param_shape = [ parameter.shape[0] * num_pth, parameter.shape[1] ]
                    merged_ckpt[parameter_name][size * i : size * (i + 1), :] = parameter
                    del parameter
                if key_to_dim[short_name] == -1:
                    size = parameter.shape[-1]
                    merged_param_shape = [ parameter.shape[0], parameter.shape[1] * num_pth]
                    merged_ckpt[parameter_name][:, size * i : size * (i + 1)] = parameter
                    del parameter
        del llama_config
    return merged_ckpt


def merge_hf_llama(cache_dir: Optional[Path] = None):
    # assert version == 2, "Only llama v2 available using huggingface"
    model = LlamaForCausalLM.from_pretrained(cache_dir, cache_dir=cache_dir, local_files_only=True, use_safetensors=False)
    weights = model.state_dict()
    weights["tok_embeddings.weight"] = weights.pop("model.embed_tokens.weight")
    weights["norm.weight"] = weights.pop("model.norm.weight")
    weights["output.weight"] = weights.pop("lm_head.weight")
    for key in list(weights.keys()):
        if rmatch := re.match(r"^model\.(layers\.[0-9]+\.)(.+)(\.weight)$", key):
            new_key = {
                "self_attn.q_proj": "attention.wq",
                "self_attn.k_proj": "attention.wk",
                "self_attn.v_proj": "attention.wv",
                "self_attn.o_proj": "attention.wo",
                "mlp.gate_proj": "feed_forward.w1",
                "mlp.down_proj": "feed_forward.w2",
                "mlp.up_proj": "feed_forward.w3",
                "input_layernorm": "attention_norm",
                "post_attention_layernorm": "ffn_norm"
            }[rmatch.group(2)]
            weights[rmatch.group(1) + new_key + rmatch.group(3)] = weights.pop(key)
    return weights, model.config

