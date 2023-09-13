import re
import sys
import os
import shutil
from pathlib import Path
from argparse import ArgumentParser

import torch
from tqdm.auto import tqdm


def permute_qkv(qkv_w: torch.Tensor, dim: int, n_heads: int,
                n_heads_kv: int, revert: bool = False) -> torch.Tensor:

    def permute(x):
        if revert:
            return x.view(head_dim//2, 2, dim).transpose(0, 1).reshape(head_dim, dim)
        return x.view(2, head_dim//2, dim).transpose(0, 1).reshape(head_dim, dim)

    head_dim = dim//n_heads
    n_qs_per_kv = n_heads//n_heads_kv
    n_groups = qkv_w.size(0)//head_dim//(n_qs_per_kv + 2)
    groups = torch.chunk(qkv_w, n_groups, dim=0)
    new = []
    for group in groups:
        *qs, k, v = torch.split(group, head_dim, dim=0)
        assert len(qs) == n_qs_per_kv, f"{len(qs)}, {n_qs_per_kv}"
        new += list(map(permute, qs)) + [permute(k), v]
    return torch.cat(new, dim=0)


def update_checkpoint(input_dir: Path, output_dir: Path, overwrite_ok: bool = False):
    # make sure megatron is importable
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     os.path.pardir)))


    # prepare output dir
    if output_dir.exists():
        if not overwrite_ok:
            raise FileExistsError(f"Output directory {output_dir} already exists")
        print(f"Removing {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(exist_ok=True)

    # determine realease
    with open(input_dir/"latest_checkpointed_iteration.txt") as f:
        it = f.read()
    print("Updating weights of iteration", it)
    with open(output_dir/"latest_checkpointed_iteration.txt", "w+") as f:
        f.write(it)
    (output_dir/it).mkdir()

    # convert weights
    for fname in tqdm(list((input_dir/it).iterdir())):
        checkpoint = torch.load(fname/"model_optim_rng.pt")
        args = checkpoint["args"]
        args = (args.hidden_size, args.num_attention_heads,
                args.num_attention_heads_kv)
        if "transformer" in checkpoint["model"]["language_model"]:
            key = "transformer"
            attn_key = "attention"
        else:
            key = "encoder"
            attn_key = "self_attention"
        states = checkpoint["model"]["language_model"][key]
        for name, weight in states.items():
            if re.match(rf"^layers\.[0-9]+\.{attn_key}\.query_key_value\.weight$", name):
                states[name] = permute_qkv(weight, *args)
        (output_dir/it/fname.stem).mkdir()
        torch.save(checkpoint, output_dir/it/fname.stem/"model_optim_rng.pt")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite-ok", action="store_true")
    args = parser.parse_args()
    update_checkpoint(args.input_dir, args.output_dir, args.overwrite_ok)
