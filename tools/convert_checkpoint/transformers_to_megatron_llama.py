import os
import sys
import shutil
from pathlib import Path
from typing import Optional
from argparse import ArgumentParser, Namespace

import torch
from tqdm.auto import trange
from transformers import AutoModelForCausalLM, LlamaTokenizer
from transformers import LlamaConfig

from permute_qkv import permute_qkv
from merge_llama import merge_hf_llama

def llama_to_megatron(weights: dict, llama_config: LlamaConfig = None) -> dict:
    def permute(qkv_w):
        return permute_qkv(qkv_w, hidden, n_heads, n_kv_heads)

    def rearrange_qkv(wq, wk, wv):
        wq = torch.split(wq, n_hidden_per_head, dim=0)
        wk = torch.split(wk, n_hidden_per_head, dim=0)
        wv = torch.split(wv, n_hidden_per_head, dim=0)
        assert len(wq) == n_heads
        assert len(wk) == n_kv_heads
        assert len(wv) == n_kv_heads
        n_qs_per_kv = n_heads//n_kv_heads
        w_qkv = []
        for i in range(n_kv_heads):
            w_qkv += [wq[i*n_qs_per_kv + j] for j in range(n_qs_per_kv)]
            w_qkv += [wk[i], wv[i]]
        return permute(torch.concat(w_qkv))

    # config
    n_layer = llama_config.num_hidden_layers
    hidden = llama_config.hidden_size
    n_heads = llama_config.num_attention_heads
    n_hidden_per_head = hidden//n_heads
    n_kv_heads = llama_config.num_key_value_heads
    # weights independent of layers
    embedding = {"word_embeddings": {"weight": weights["tok_embeddings.weight"]}}
    transformer = {"final_layernorm.weight": weights["norm.weight"]}
    lm_head = weights["output.weight"]
    # get all the other weights
    for layer in trange(n_layer, desc="Converting weights"):
        prefix = f"layers.{layer}"
        # identical weights
        transformer[f"{prefix}.attention.dense.weight"] = \
            weights[f"{prefix}.attention.wo.weight"]
        transformer[f"{prefix}.post_attention_layernorm.weight"] = \
            weights[f"{prefix}.ffn_norm.weight"]
        transformer[f"{prefix}.input_layernorm.weight"] = \
            weights[f"{prefix}.attention_norm.weight"]
        transformer[f"{prefix}.mlp.dense_4h_to_h.weight"] = \
            weights[f"{prefix}.feed_forward.w2.weight"]
        # concatenate up, gate mlp weights
        transformer[f"{prefix}.mlp.dense_h_to_4h.weight"] = torch.concat([
            weights[f"{prefix}.feed_forward.w3.weight"],
            weights[f"{prefix}.feed_forward.w1.weight"]
        ])
        # finally, qkv requires serious manipulation to get right
        transformer[f"{prefix}.attention.query_key_value.weight"] = rearrange_qkv(
            weights[f"{prefix}.attention.wq.weight"],
            weights[f"{prefix}.attention.wk.weight"],
            weights[f"{prefix}.attention.wv.weight"]
        )

        # release references to original weights (free mem)
        del weights[f"{prefix}.feed_forward.w3.weight"]
        del weights[f"{prefix}.feed_forward.w1.weight"]
        del weights[f"{prefix}.attention.wq.weight"]
        del weights[f"{prefix}.attention.wk.weight"]
        del weights[f"{prefix}.attention.wv.weight"]

    return {"embedding": embedding, "encoder": transformer,
            "lm_head": lm_head}

def main(out: Optional[Path] = None,
         cache_dir: Optional[Path] = None, megatron_path: Optional[Path] = None):

    if megatron_path:
        print("Add megatron to os path")
        os.path.append(megatron_path)
    # get weights from or specified directory
    print("Getting llama...")
    hf_weights, llama_config = merge_hf_llama(cache_dir)

    # convert state dict to be megatron-compatible
    megatron_weights = llama_to_megatron(hf_weights, llama_config=llama_config)

    # set args
    # llama1, llama2
    args = {"num_layers": llama_config.num_hidden_layers,
            "hidden_size": llama_config.hidden_size,
            "num_attention_heads": llama_config.num_attention_heads,
            "ffn_hidden_size": llama_config.intermediate_size,
            "num_key_value_heads": llama_config.num_key_value_heads,
            "parallel_attn": False,
            "make_vocab_size_divisible_by": 1,
            "glu_activation": "swiglu",
            "max_position_embeddings": llama_config.max_length, # should use max_length rather than max_position_embeddings, detail in https://github.com/lm-sys/FastChat/issues/2046#issuecomment-1645265800
            "seq_length": llama_config.max_length, 
            "layernorm_epsilon": llama_config.rms_norm_eps,
            # llama args
            "padded_vocab_size": llama_config.vocab_size,
            "tokenizer_type": "GPTSentencePieceTokenizer",
            "no-query-key-layer-scaling": True,
            "attention-dropout": 0,
            "hidden-dropout": 0,
            "use-rotary-position-embeddings": True,
            "untie-embeddings-and-output-weights": True,
            "swiglu": True,
            "normalization": "rmsnorm",
            "disable-bias-linear": True,
            "add_position_embedding": False,
            "add_bias_linear": False,
            }
    if llama_config.num_key_value_heads:
        args.update({"num_attention_heads_kv": llama_config.num_key_value_heads})

    args.update({
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "iteration": 0,
        "bias_gelu_fusion": False,
        "bias_droput_fusion": False,
    })

    # save converted weights in specified out
    (out/"release"/"mp_rank_00").mkdir(parents=True)
    with open(out/"latest_checkpointed_iteration.txt", "w+") as f:
        f.write("release")
    final_dict = {"iteration": 'release', "model": {"language_model": megatron_weights},
                  "checkpoint_version": 3.0, "args": Namespace(**args)}
    torch.save(final_dict, out/"release"/"mp_rank_00"/"model_optim_rng.pt")
    print("Saved weights in", out)

    tokenizer = LlamaTokenizer.from_pretrained(
        cache_dir, cache_dir=cache_dir, local_files_only=True,
    )
    token_path = out/"tokenizer.model"
    vocab_file = tokenizer.vocab_file
    shutil.copy(vocab_file, token_path)
    print("Saved tokenizer.model in", token_path)
    print("Done")

if __name__ == "__main__":
    parser = ArgumentParser(description="Convert Huggingface llama weights to "
                                        "megatron-compatible weights")
    parser.add_argument("--out", type=Path,
                        help="Directory to store the megatron weights (as checkpoint)")
    parser.add_argument("--cache-dir", type=Path,
                        help=("Directory to store the huggingface weights, or "
                              "in case of the llama model, where to look for "
                              "the consolidated.xx.pth"))
    parser.add_argument("--megatron-path", type=Path, default=None,
                        help="Path where to find megatron code")
    args = parser.parse_args()

    main(args.out, args.cache_dir, args.megatron_path)
