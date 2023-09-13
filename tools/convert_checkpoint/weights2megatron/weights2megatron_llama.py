import os
import sys
import shutil
from pathlib import Path
from typing import Optional
from argparse import ArgumentParser, Namespace

import torch
from tqdm.auto import trange
from transformers import AutoModelForCausalLM, LlamaTokenizer

from permute_qkv import permute_qkv
from merge_llama import merge_llama
from transformers import AutoTokenizer

llama_s2layer = {7: 32, 13: 40, 30: 60, 65: 80, 70: 80}
llama_s2heads = {7: 32, 13: 40, 30: 52, 65: 64, 70: 64}
llama_s2dense = {7: 11008, 13: 13824, 30: 17920, 65: 22016,
                 70: 28672}  # should be (2/3)*4*d, but it isn't exaclty that
llama_s2hidden = {7: 4096, 13: 5120, 30: 6656, 65: 8192, 70: 8192}


def llama_to_megatron(weights: dict, size: int, source: str = "meta",
                      version: int = 1) -> dict:
    def permute(qkv_w):
        if source == "hf":
            return permute_qkv(qkv_w, hidden, n_heads, n_kv_heads)
        return qkv_w

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
    n_layer = llama_s2layer[size]
    hidden = llama_s2hidden[size]
    n_heads = llama_s2heads[size]
    n_hidden_per_head = hidden//n_heads
    n_kv_heads = n_heads if version == 1 or size <= 13 else 8

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

def main(model_name: str = "llama2", size: int = 7, out: Optional[Path] = None,
         cache_dir: Optional[Path] = None, megatron_path: Optional[Path] = None, padded_vocab_size: Optional[int] = 32000):

    # get weights from or specified directory
    print("Getting llama...")
    version = 2 if "2" in model_name else 1
    hf_weights, llama_source = merge_llama(size, version, cache_dir, padded_vocab_size)

    # convert state dict to be megatron-compatible
    megatron_weights = llama_to_megatron(hf_weights, size, llama_source,
                                            version=1 if model_name == "llama" else 2)

    # set args
    # llama1, llama2
    args = {"num_layers": llama_s2layer[size],
            "hidden_size": llama_s2hidden[size],
            "num_attention_heads": llama_s2heads[size],
            "ffn_hidden_size": llama_s2dense[size],
            "num_key_value_heads": llama_s2heads[size],
            "parallel_attn": False,
            "make_vocab_size_divisible_by": 1,
            "glu_activation": "swiglu",
            # llama args
            "padded_vocab_size": padded_vocab_size,
            "use_rms_norm": True,
            "tie_embed_logits": False,
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
    if model_name == "llama":
        args.update({"max_position_embeddings": 2048, "seq_length": 2048,
                        "layernorm_epsilon": 1e-6})
    else:  # llama2
        args.update({"max_position_embeddings": 2048, "seq_length": 2048,
                        "layernorm_epsilon": 1e-5})
        if size >= 34:
            args.update({"num_attention_heads_kv": 8})

    args.update({
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "iteration": "release",
        "bias_gelu_fusion": False,
        "bias_droput_fusion": False,
    })

    # save converted weights in specified out
    (out/"release"/"mp_rank_00").mkdir(parents=True)
    with open(out/"latest_checkpointed_iteration.txt", "w+") as f:
        f.write("release")
    final_dict = {"iteration": "release", "model": {"language_model": megatron_weights},
                  "checkpoint_version": 3.0, "args": Namespace(**args)}
    torch.save(final_dict, out/"release"/"mp_rank_00"/"model_optim_rng.pt")
    print("Saved weights in", out)

    if model_name == "llama2" and llama_source == "hf":
        tokenizer = LlamaTokenizer.from_pretrained(
            cache_dir, cache_dir=cache_dir, local_files_only=True,
        )
        token_path = out/"tokenizer.model"
        vocab_file = tokenizer.vocab_file
        shutil.copy(vocab_file, token_path)
        print("Saved tokenizer.model in", token_path)
    print("Done")

if __name__ == "__main__":
    parser = ArgumentParser(description="Convert Huggingface falcon weights to "
                                        "megatron-compatible weights")
    parser.add_argument("model", choices={"falcon", "llama", "llama2"})
    parser.add_argument("--size", default=7, choices={7, 13, 30, 34, 40, 65, 70}, type=int,
                        help="The size of the model")
    parser.add_argument("--out", type=Path,
                        help="Directory to store the megatron weights (as checkpoint)")
    parser.add_argument("--cache-dir", type=Path,
                        help=("Directory to store the huggingface weights, or "
                              "in case of the llama model, where to look for "
                              "the consolidated.xx.pth"))
    parser.add_argument("--megatron-path", type=Path,
                        help="Path where to find megatron code")
    parser.add_argument("--tokenizer-size", type=int, help="Directory to store the megatron weights (as checkpoint)", default=None)
    args = parser.parse_args()

    # small arg verification
    if args.model == "llama":
        assert args.size in {7, 13, 30, 65}
    else:
        assert args.size in {7, 13, 70}

    main(args.model, args.size, args.out, args.cache_dir, args.megatron_path, args.tokenizer_size)
