#! /bin/bash

# Runs the "345M" parameter model
#export MUDNN_LOG_LEVEL=INFO
export DS_ACCELERATOR=musa
export MTHREADS_VISIBLE_DEVICES=all
export PYTHONPATH=$PATHONPATH:/workspace/LLM
#export DISTRIBUTED_BACKTRACE=1
GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=/data/dataset/BookCorpusDataset_text_document
CHECKPOINT_PATH=/data/dataset/checkpoints/gpt2_345m

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

torchrun $DISTRIBUTED_ARGS \
       ../pretrain_gpt.py \
       --tensor-model-parallel-size 1 \
       --deepspeed \
       --deepspeed_config ./ds_config.json \
       --no-bias-dropout-fusion \
       --no-masked-softmax-fusion \
       --no-bias-gelu-fusion \
       --no-gradient-accumulation-fusion \
       --num-layers 2 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --micro-batch-size 1 \
       --global-batch-size 4 \
       --seq-length 1024 \
       --max-position-embeddings 2048 \
       --train-iters 500000 \
       --lr-decay-iters 320000 \
       --data-path $DATA_PATH \
       --vocab-file /data/dataset/gpt2-vocab.json \
       --merge-file /data/dataset/gpt2-merges.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend mccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 0.0 \
       --lr-warmup-fraction .01 \
       --log-interval 1 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10  \
       --prometheus-gateway "192.168.41.156:9091" \
       --service gpt-pretrain-deepspeed