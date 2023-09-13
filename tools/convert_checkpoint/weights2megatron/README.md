# Introduction
This folder is a collection of scripts for converting hf checkpoints to megatron-DeepSpeed checkpoints.

# Usage
## huggingface to megatron 
```bash
python weights2megatron/weights2megatron.py llama2 --size=13 --out=${DEST_DIR} --cache-dir=${HF_CKPT_DIR} --tokenizer-size=32000
```

## split ckpt by TP and PP size
```bash
 python3 tools/checkpoint_util.py   \
       --target-tensor-parallel-size 4 \
       --target-pipeline-parallel-size 2 \
       --load-dir ${LOAD_DIR}   \
       --save-dir ${SAVE_DIR}  \
       --model-type GPT  \
       --true-vocab-size 32000
```