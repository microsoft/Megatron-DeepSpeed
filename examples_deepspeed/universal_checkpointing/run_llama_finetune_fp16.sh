DS_CONFIG=examples_deepspeed/finetune_hf_llama/ds_config.json
DATASET_PATH=./dataset/alpaca_data.json
# dataset link: https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json

HF_LLAMA_PATH=./dataset/llama-7b/
# weights link: https://huggingface.co/huggyllama/llama-7b

# require to align with weight dimensions
HIDDEN_SIZE=4096
FFN_HIDDEN_SIZE=11008
NUM_LAYERS=32
NUM_HEADS=32
SEQ_LENGTH=512

# TODO (lekurile): Bring over UC args here
SIZE_TAG="toy"
ZERO_STAGE=1
DTYPE="fp16"

## Debug
#DEBUG_MODE=1
#if [[ $DEBUG_MODE == 1 ]]; then
#        LAYERS=4
#        HIDDEN=512
#        SEQ=512
#        EXIT_INTERVAL=200
#else
#        HIDDEN=1024
#        LAYERS=24
#        SEQ=1024
#        EXIT_INTERVAL=100
#        SIZE_TAG="big"
#fi


# 3D parallelism of training
TP=2
PP=2
DP=2
SP=1
#MICRO_BATCH_SIZE=16
#GLOBAL_BATCH_SIZE=256
WORLD_SIZE=$((TP*PP*DP*SP))
GLOBAL_BATCH=256
MICRO_BATCH=$((GLOBAL_BATCH/WORLD_SIZE))
TRAIN_ITERS=3500
LR=2e-5

DEBUG_MODE=1
if [[ $DEBUG_MODE == 1 ]]; then
    EXIT_INTERVAL=200
else
    EXIT_INTERVAL=$TRAIN_ITERS
fi

# 3D parallelism of checkpoint to load
LOAD_TP=$TP
LOAD_PP=$PP
LOAD_DP=$DP
LOAD_SP=$SP
RUN_TAG="save"

######################################

# Below configuration required for llama model as per llama paper
# --no-query-key-layer-scaling \
# --attention-dropout 0 \
# --hidden-dropout 0 \
# --use-rotary-position-embeddings \
# --untie-embeddings-and-output-weights \
# --swiglu \
# --normalization rmsnorm \
# --disable-bias-linear \
######################################
cat <<EOT > $DS_CONFIG
{
  "train_batch_size" : $GLOBAL_BATCH,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH,
  "steps_per_print": 1,

  "zero_optimization": {
    "stage": $ZERO_STAGE
  },

  "bf16": {
    "enabled": false
  },

  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 50,
    "hysteresis": 2,
    "min_loss_scale": 0,
    "initial_scale_power": 12
  },

  "wall_clock_breakdown" : false
}
}
EOT

#---------------------------------------------
#   TODO (lekurile): Test code
#---------------------------------------------
MEGA_DS_LLAMA_PATH=./"llama_7b_mega_ds_tp${TP}_pp${PP}_dp${DP}_sp${SP}_${size_tag}"

EXP_DIR="z${ZERO_STAGE}_uni_ckpt"
CHECKPOINT_PATH=${EXP_DIR}/checkpoints/llama/z${ZERO_STAGE}/$DTYPE/tp${TP}_pp${PP}_dp${DP}_sp${SP}_${SIZE_TAG}
LOAD_CHECKPOINT_PATH=${EXP_DIR}/checkpoints/llama/z${ZERO_STAGE}/$DTYPE/tp${LOAD_TP}_pp${LOAD_PP}_dp${LOAD_DP}_sp${LOAD_SP}_${SIZE_TAG}
LOG_DIR="${EXP_DIR}/tensorboard/$DTYPE/tp${TP}_pp${PP}_dp${DP}_sp${SP}_hd${HIDDEN}_nl${LAYERS}_gbsz${GLOBAL_BATCH}_mbsz${MICRO_BATCH}_z${ZERO_STAGE}_LR_${LR}_${DTYPE}_${SIZE_TAG}_${RUN_TAG}"
mkdir -p $LOG_DIR

covert_args="deepspeed tools/hf2megads_weight_converter.py \
--hf-ckpt-num-shards 2 \
--origin-hf-ckpt-dir $HF_LLAMA_PATH \
--save $MEGA_DS_LLAMA_PATH"

finetune_args="deepspeed finetune_llama.py \
--load $MEGA_DS_LLAMA_PATH \
--save $CHECKPOINT_PATH"

comm_args="--tensor-model-parallel-size $TP \
--pipeline-model-parallel-size $PP \
--ds-sequence-parallel-size $SP \
--lr-warmup-iters 2000 \
--weight-decay 0.1 \
--clip-grad 1 \
--num-layers $NUM_LAYERS \
--hidden-size $HIDDEN_SIZE \
--num-attention-heads $NUM_HEADS \
--ffn-hidden-size $FFN_HIDDEN_SIZE \
--attention-dropout 0 \
--hidden-dropout 0 \
--no-query-key-layer-scaling \
--disable-bias-linear \
--normalization rmsnorm \
--use-rotary-position-embeddings \
--untie-embeddings-and-output-weights \
--swiglu \
--seq-length $SEQ_LENGTH \
--max-position-embeddings $SEQ_LENGTH \
--micro-batch-size $MICRO_BATCH \
--global-batch-size $GLOBAL_BATCH \
--train-iters $TRAIN_ITERS \
--lr $LR \
--tensorboard-dir $LOG_DIR \
--lr-decay-iters 320000 \
--lr-decay-style cosine \
--log-interval 1 \
--eval-iters 40 \
--eval-interval 10 \
--data-path $DATASET_PATH \
--save-interval 100 \
--split 100,0,0 \
--${DTYPE} \
--zero-stage=${ZERO_STAGE} \
--tokenizer-type HFTokenizer \
--tokenizer-model $HF_LLAMA_PATH \
--deepspeed_config $DS_CONFIG \
--deepspeed \
--distributed-backend nccl \
--num-workers 0 \
--no-masked-softmax-fusion \
--no-bias-gelu-fusion \
--no-bias-dropout-fusion \
--no-gradient-accumulation-fusion \
--exit-interval ${EXIT_INTERVAL} \
--repeated-dataloader"


if [[ ${ZERO_STAGE} -gt 1 ]]; then
comm_args="${comm_args} \
    --no-pipeline-parallel"
fi

if [ "$1" = "convert" ]; then
    task_args="$covert_args"
else
    task_args="$finetune_args"
fi

full_cmd="$task_args $comm_args"

#eval "$full_cmd"

#---------------------------------------------
#   TODO (lekurile): Test code
#---------------------------------------------
echo ${comm_args}
echo ${$full_cmd}
eval ${$full_cmd}

set +x
