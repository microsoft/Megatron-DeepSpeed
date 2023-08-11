#!/bin/bash

# Runs the "345M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
# Change for multinode config
export DS_ACCELERATOR=musa
export MTHREADS_VISIBLE_DEVICES=all
export PYTHONPATH=$PATHONPATH:/workspace/LLM
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

#CHECKPOINT_PATH=<Specify path>
VOCAB_FILE=/workspace/Megatron-DeepSpeed/dataset/gpt2-vocab.json
MERGE_FILE=/workspace/Megatron-DeepSpeed/dataset/gpt2-merges.txt
DATA_PATH=/workspace/Megatron-DeepSpeed/dataset/BookCorpusDataset_text_document

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 2 \
    --sequence-parallel \
    --no-bias-dropout-fusion \
    --no-masked-softmax-fusion \
    --no-bias-gelu-fusion \
    --no-gradient-accumulation-fusion \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --decoder-seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 4 \
    --global-batch-size 16 \
    --lr 0.00015 \
    --train-iters 500000 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 0.0
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --data-impl mmap \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
"

torchrun $DISTRIBUTED_ARGS /workspace/all/Megatron-DeepSpeed/pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend mccl \
    --prometheus-gateway "192.168.41.156:9091" \
    --service gpt-pretrain-deepspeed
   # --save $CHECKPOINT_PATH \
   # --load $CHECKPOINT_PATH