#!/bin/bash
# This example script is contributed by external user https://github.com/nrailgun
# [2023-12-20]: Modified by [@saforem2](https://github.com/saforem2)
# set -ex
#
function WhereAmI() {
    python3 -c 'import os; print(os.getcwd())'
}

function join_by { local d=${1-} f=${2-}; if shift 2; then printf %s "$f" "${@/#/$d}"; fi; }

function sourceFile() {
    FILE="$1"
    echo "source-ing ${FILE}"
    if [[ -f "${FILE}" ]]; then
        # shellcheck source="${FILE}"
        source "${FILE}"
    else
        echo "ERROR: UNABLE TO SOURCE ${FILE}"
    fi
}


USER=$(whoami)
HERE=$(WhereAmI)
ALCF_DIR="${HERE}/ALCF"
PARENT=$(dirname "${ALCF_DIR}")
echo "+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+"
echo "ALCF_DIR: ${ALCF_DIR}"
echo "PARENT: ${PARENT}"
echo "+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+"


HOSTNAME=$(hostname)
sourceFile "${ALCF_DIR}/setup.sh"
sourceFile "${ALCF_DIR}/model.sh"

WORLD_SIZE="${NGPUS}"
PARALLEL_SIZE="${WORLD_SIZE}"
echo "NHOSTS * (NGPU / HOST) = $NHOSTS * $NGPU_PER_HOST = $NGPUS"

# MODEL_LLAMA_KEY="LLAMA-24L"
# HIDDEN_SIZE=2048 # e.g. llama-13b: 5120
# FFN_HIDDEN_SIZE=5504 # e.g. llama-13b: 13824
# NUM_LAYERS=24 # e.g. llama-13b: 40
# NUM_HEADS=16 # e.g. llama-13b: 40
# SEQ_LENGTH=2048
# NUM_KV_HEADS=4 # llama2 70B uses GQA
# FFN_HIDDEN_SIZE=5504
# NUM_HEADS=16 # e.g. llama-13b: 40
######################################
# Change the below configurations here
# wget https://the-eye.eu/public/AI/pile_neox/data/BookCorpusDataset_text_document.bin
# wget https://the-eye.eu/public/AI/pile_neox/data/BookCorpusDataset_text_document.idx

USER=$(whoami)
HERE=$(WhereAmI)
ALCF_DIR="${HERE}/ALCF"
PARENT=$(dirname "${ALCF_DIR}")
# echo "+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+"
# echo "ALCF_DIR: ${ALCF_DIR}"
# # echo "PARENT: ${PARENT}"
# echo "+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+"

MEGATRON_DIR="${HERE}"

# DATA_DIR="${HOME}/datascience/foremans/locations/thetaGPU/projects/saforem2/Megatron-DeepSpeed/dataset/"
BASE_PATH="${MEGATRON_DIR}"
DS_CONFIG=${BASE_PATH}/deepspeed.json
DATASET_1="${DATA_DIR}/BookCorpusDataset_text_document"
# DATASET_1="./tmp/data/bookcorpus_train_1m_text_sentence"
DATASET="1 ${DATASET_1}"
# CHECKPOINT_PATH=./tmp
TOKENIZER_PATH=./tmp/tokenizer.model # offical llama tokenizer.model

if [[ $(hostname) == nid* || $(hostname) == login* ]]; then
    DATA_PARENT="/global/homes/f/foremans/m3957/foremans/projects/saforem2/Megatron-DeepSpeed"
    DATA_TYPE="BookCorpusDataset_text_document"
elif [[ $(hostname) == theta* || $(hostname) == x3* ]]; then
    DATA_PARENT="/lus/grand/projects/fallwkshp23/datasets/GenSLMSubSample200k"
    DATA_TYPE="genslm_subsample_200k_sequence_document"
else
    echo "Unable to determine DATA_PARENT for $(hostname)."
    echo "Exiting!"
    exit 1
fi

DATA_DIR="${DATA_PARENT}/dataset"
DATA_PATH="${DATA_DIR}/${DATA_TYPE}"
VOCAB_FILE="${DATA_DIR}/gpt2-vocab.json"
MERGE_FILE="${DATA_DIR}/gpt2-merges.txt"

echo "+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+"
echo "ALCF_DIR: ${ALCF_DIR}"
echo "MEGATRON_DIR: ${MEGATRON_DIR}"
echo "+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+"

DATA_LOAD_ARGS=(
    "--data-path $DATA_PATH"
    "--vocab-file $VOCAB_FILE"
    "--merge-file $MERGE_FILE"
)

# TP=2
# PP=2
# ZERO_STAGE=0

GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=$(wc -l < "${PBS_NODEFILE:-${COBALT_NODEFILE:-1}}")
NODE_RANK=0

# TP=2
# PP=2
# ZERO_STAGE=0
#
export SEQ_LENGTH=${SEQ_LENGTH:-2048}
export NUM_KV_HEADS=4 # llama2 70B uses GQA
export MODEL_SIZE_KEY="${MODEL_SIZE_KEY:-LLAMA_7B}"
export MODEL_TYPE=${MODEL_TYPE:-llama}
echo "==========================+"
echo "Using ${MODEL_SIZE_KEY}"
echo "==========================+"


export DDP_IMPL="local"
export GAS=${GAS:-1}
export MPSIZE=${MPSIZE:-1}
export SPSIZE=${SPSIZE:-1}
export PPSIZE=${PPSIZE:-1}
export SP_TYPE=${SP_TYPE:-"ds"}
export MICRO_BATCH=${MICRO_BATCH:-1}

# export HIDDEN_SIZE=2048 # e.g. llama-13b: 5120
# export FFN_HIDDEN_SIZE=5504 # e.g. llama-13b: 13824
# export NUM_LAYERS=24 # e.g. llama-13b: 40
# export NUM_HEADS=16 # e.g. llama-13b: 40
# export SEQ_LENGTH=${SEQ_LENGTH:-2048}
# export NUM_KV_HEADS=4 # llama2 70B uses GQA

NUM_KV_HEADS=4 # llama2 70B uses GQA
FFN_HIDDEN_SIZE=5504

# GLOBAL_BATCH=32 # e.g. llama: 4M tokens
TRAIN_STEPS=250000 # e.g. llama: 1T tokens / 4M tokens_per_batch = 250000 steps
LR=3e-4
MIN_LR=3e-5
LR_WARMUP_STEPS=2000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

## Activation checkpointing saves GPU memory, but reduces training speed
# activation_checkpoint="true"
activation_checkpoint="false"

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

# Deal with Sequence Parallel implementation ---------------------------------------
# ----------------------------------------------------------------------------------
if [[ ${SP_TYPE} == "ds" ]]; then
    # NOTE: --------------------------------------------------------------------
    # SP_TYPE="ds" has NO effect, essentially running with no Seq. parallelism
    # --------------------------------------------------------------------------
    if [[ "$MPSIZE" == "${WORLD_SIZE}" ]]; then
        # hacky workaround to try and use SP_TYPE="ds" + MPSIZE="${WORLD_SIZE}"
        # ------------------------------------------------------------------------
        # Update [2023-08-22]: Chengming mentioned that this is an internal issue
        # and will NOT work currently
        # ------------------------------------------------------------------------
        echo "Caught MPSIZE: $MPSIZE from env. Setting SPSIZE=1"
        SPSIZE=1
        MPSIZE="${MPSIZE}"
    else
        echo "Didn't catch MPSIZE from env. Setting SPSIZE=${WORLD_SIZE}, MPSIZE=1"
        MPSIZE=1
        SPSIZE="${WORLD_SIZE}"
    fi
    if [ -z "${ZERO_STAGE}" ]; then
        echo "ZERO_STAGE not set, setting to 3 for ${SP_TYPE}"
        ZERO_STAGE=3
    else
        echo "Caught ZERO_STAGE=${ZERO_STAGE} with ${SP_TYPE}"
    fi
    export SPSIZE="${SPSIZE:-$WORLD_SIZE}"
    export MPSIZE="${MPSIZE:-1}"
    export USE_SEQUENCE_PARALLEL=0
    export ZERO_STAGE="${ZERO_STAGE}"
elif [[ ${SP_TYPE} == "megatron" ]]; then
    # NOTE: --------------------------------------------------------------------------
    # SP_TYPE="megatron" will use Megatron's Seq. || implementation with ZERO_STAGE=0
    # --------------------------------------------------------------------------------
    [ "$SPSIZE" ] && echo "Caught SPSIZE: ${SPSIZE} from env" || SPSIZE=1
    [ "$MPSIZE" ] && echo "Caught MPSIZE: ${MPSIZE} from env" || MPSIZE="${WORLD_SIZE}"
    [ "$ZERO_STAGE" ] && echo "Caught ${ZERO_STAGE} from env" || ZERO_STAGE=0
    [ "$USE_SEQUENCE_PARALLEL" ] && echo "Caught USE_SP: $USE_SEQUENCE_PARALLEL from env" || USE_SEQUENCE_PARALLEL=1
    if [[ ${PPSIZE} > 1 ]]; then # && ${MPSIZE}==${WORLD_SIZE} ]];
        MPSIZE=$(( WORLD_SIZE / PPSIZE ))
        echo "Re-setting MPSIZE to ${WORLD_SIZE} / ${PPSIZE} = $(( WORLD_SIZE / PPSIZE ))"
        echo "MPSIZE: $MPSIZE"
        # MPSIZE="${WORLD_SIZE}/"
    fi
    export SPSIZE="${SPSIZE}"
    export MPSIZE="${MPSIZE}"
    export ZERO_STAGE="${ZERO_STAGE}"
    export USE_SEQUENCE_PARALLEL="${USE_SEQUENCE_PARALLEL:-1}"
else
    echo "Unexpected SP_TYPE: ${SP_TYPE}"
    # exit 1
fi
# ------------------------------------------------------------------------
#
echo "####################################################"
echo "USING: ${SP_TYPE}" 
echo "SPSIZE: ${SPSIZE}"
echo "PPSIZE: ${SPSIZE}"
echo "MPSIZE: ${MPSIZE}"
echo "ZERO_STAGE: ${ZERO_STAGE}"
echo "WORLD_SIZE: ${WORLD_SIZE}"
echo "USE_SEQUENCE_PARALLEL: ${USE_SEQUENCE_PARALLEL}"
echo "####################################################"

echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "${SP_TYPE} sequence parallelism, with: "
echo "    {MPSIZE: ${MPSIZE}, SPSIZE: ${SPSIZE}, USE_SEQUENCE_PARALLEL: ${USE_SEQUENCE_PARALLEL}} !!"
echo "########################################################"

GLOBAL_BATCH=$(( NGPUS * MICRO_BATCH * GAS ))

GLOBAL_BATCH=$(( GLOBAL_BATCH / MPSIZE / PPSIZE / SPSIZE))

echo "GB = (NGPUS * MB * GAS) / (MP * PP * SP * DP) = ${NGPUS} * ${MICRO_BATCH} * ${GAS} = ${GLOBAL_BATCH} / (${MPSIZE} * ${PPSIZE} * ${PPSIZE})"
# echo "GB = (NGPUS * MB * GAS) / (MP * PP * SP) = (${NGPUS} * ${MICRO_BATCH} * ${GAS}) / (${MPSIZE} * ${PPSIZE} * ${SPSIZE}) = ${GLOBAL_BATCH}"

if [[ "${GLOBAL_BATCH}" == 0 ]]; then
    GLOBAL_BATCH=1
fi
# [ "${GLOBAL_BATCH:-${GLOBAL_BATCH}}" == 0 ] && GLOBAL_BATCH=1 || echo "GLOBAL_BATCH: ${GLOBAL_BATCH}"

DPSIZE=$(( $WORLD_SIZE / $PPSIZE / $MPSIZE ))

export GLOBAL_BATCH="$(( GLOBAL_BATCH * DPSIZE ))"
# export GLOBAL_BATCH="$GLOBAL_BATCH"
# echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
# echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"

echo "--------------------------------"
echo "GLOBAL_BATCH=${GLOBAL_BATCH}"
echo "USING DPSIZE: ${DPSIZE}"
echo "--------------------------------"

# REMAINDER=$(( GLOBAL_BATCH % (MICRO_BATCH * DPSIZE)))
# if [[ "${GLOBAL_BATCH} "]]




cat <<EOT > $DS_CONFIG
{
  "train_batch_size" : $GLOBAL_BATCH,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH,
  "steps_per_print": 1,
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "bf16": {
    "enabled": true
  },
  "wandb": {
    "enabled": true,
    "project": "GenSLM-Megatron-DS"
  }
}
EOT

ds_args=""
ds_args=" --deepspeed ${ds_args}"
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"

if [ "${activation_checkpoint}" = "true" ]; then
  ds_args="--deepspeed-activation-checkpointing ${ds_args}"

  ## old argument for recomputing the transformer layer
  # ds_args="--checkpoint-activations ${ds_args}"

  ## new argument for recomputing the transformer layer
  ds_args="--recompute-granularity full --recompute-method uniform ${ds_args}"
  ## new argument for recomputing only the attention layer
  # ds_args="--recompute-granularity selective ${ds_args}"
fi


DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

# torchrun $DISTRIBUTED_ARGS \
#        pretrain_gpt.py \


# ┏━━━━━━━━━━━━━━━━━━━┓
# ┃ FILE I/O SETTINGS ┃
# ┗━━━━━━━━━━━━━━━━━━━┛
RUN_STR="gb${GLOBAL_BATCH}_mb${MICRO_BATCH}"
RUN_STR="nl${NLAYERS}_hs${HIDDEN}_${RUN_STR}"
RUN_STR="mp${MPSIZE}_pp${PPSIZE}_sp${SPSIZE}_${RUN_STR}"
RUN_STR="z${ZERO_STAGE}_seqlen${SEQ_LEN}_${RUN_STR}"
RUN_STR="${MODEL_SIZE_KEY}_${RUN_STR}"

# if [[ "${USE_FLASH_ATTN}" == 0 ]]; then
#     echo "Not using Flash Attention!!"
# else
#
if [[ "${USE_FLASH_ATTN1}" || "${USE_FLASH_ATTN_V1}" ]]; then
    # Flash Attention 1
    [ "${USE_FLASH_ATTN}" ] && RUN_STR="flashAttn_v1_${RUN_STR}"
    [ "${USE_FLASH_ATTN1}" ] && RUN_STR="flashAttn_v1_${RUN_STR}"
    [ "${USE_FLASH_ATTN_V1}" ] && RUN_STR="flashAttn_v1_${RUN_STR}"
elif [[ "${USE_FLASH_ATTN2}" || "${USE_FLASH_ATTN_V2}" ]]; then
    # Flash Attention 2
    [ "${USE_FLASH_ATTN2}" ] && RUN_STR="flashAttn_v2_${RUN_STR}"
    [ "${USE_FLASH_ATTN_V2}" ] && RUN_STR="flashAttn_v2_${RUN_STR}"
elif [[ "${USE_FLASH_ATTN_TRITON}" ]]; then
    # Triton + Flash Attn
    # Triton + Flash Attn
    [ "${USE_FLASH_ATTN_TRITON}" ] && RUN_STR="flashAttn_triton_${RUN_STR}"
else
    echo "Not using Flash Attention!"
fi

if [[ $DDP_IMPL == 'FSDP' ]]; then
    RUN_STR="FSDP_${RUN_STR}"
fi
if [[ $USE_ACTIVATION_CHECKPOINTING == 1 ]]; then
    RUN_STR="actCkpt_${RUN_STR}"
fi
if [[ $USE_SEQUENCE_PARALLEL == 1 ]] ; then
    RUN_STR="SP_${RUN_STR}"
fi

RUN_STR="${MODEL_SIZE}_${RUN_STR}"

OUTPUT_DIR="${PARENT}/outputs/${RUN_STR}"
CHECKPOINT_DIR="${PARENT}/checkpoints/$RUN_STR"
TENSORBOARD_DIR="${PARENT}/outputs/${RUN_STR}/tensorboard"

DATE=$(date)
export DATE="${DATE}"
export RUN_STR="${RUN_STR}"
export MODEL_SIZE="${MODEL_SIZE:-${MODEL_SIZE_KEY}}"
export MODEL_SIZE="$MODEL_SIZE"
export TENSORBOARD_DIR=$TENSORBOARD_DIR
export OUTPUT_DIR=$OUTPUT_DIR
mkdir -p "$OUTPUT_DIR/tensorboard/wandb"
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$TENSORBOARD_DIR"
mkdir -p "$OUTPUT_DIR"
echo "OUTPUT TO: ${OUTPUT_DIR}"

gpt_args=(
    "--tensor-model-parallel-size $MPSIZE"
    "--pipeline-model-parallel-size $PPSIZE"
    "--num-layers $NLAYERS"
    "--hidden-size $HIDDEN"
    "--ffn-hidden-size $FFN_HIDDEN_SIZE"
    "--num-attention-heads $ATEN_HEADS"
    "--micro-batch-size $MICRO_BATCH"
    "--global-batch-size $GLOBAL_BATCH"
    "--seq-length $SEQ_LENGTH"
    "--max-position-embeddings $SEQ_LENGTH"
    "--train-iters $TRAIN_STEPS"
    "--save $CHECKPOINT_DIR"
    "--load $CHECKPOINT_DIR"
    "--data-path $DATASET"
    "--data-impl mmap"
    "--tokenizer-type GPTSentencePieceTokenizer"
    "--tokenizer-model $TOKENIZER_PATH"
    "--split 949,50,1"
    "--distributed-backend nccl"
    "--lr $LR"
    "--lr-decay-style cosine"
    "--min-lr $MIN_LR"
    "--weight-decay $WEIGHT_DECAY"
    "--clip-grad $GRAD_CLIP"
    "--lr-warmup-iters $LR_WARMUP_STEPS"
    "--optimizer adam"
    "--adam-beta1 0.9"
    "--adam-beta2 0.95"
    "--log-interval 1"
    "--save-interval 10000"
    "--eval-interval 1000"
    "--eval-iters 10"
    "--bf16"
    "--no-query-key-layer-scaling"
    "--attention-dropout 0"
    "--hidden-dropout 0"
    "--use-rotary-position-embeddings"
    "--untie-embeddings-and-output-weights"
    "--swiglu"
    "--normalization rmsnorm"
    "--disable-bias-linear"
    "--num-key-value-heads $NUM_KV_HEADS"
    "--tensorboard-dir ${TENSORBOARD_DIR}"
    "--log-timers-to-tensorboard"
    "--tensorboard-log-interval 1"
    "--data-path $DATA_PATH"
    "--vocab-file $VOCAB_FILE"
    "--merge-file $MERGE_FILE"
)

# DATA_LOAD_ARGS=(
#     "--data-path $DATA_PATH"
#     "--vocab-file $VOCAB_FILE"
#     "--merge-file $MERGE_FILE"
# )

export gpt_args=(
    "${gpt_args[*]}"
    "${ds_args[*]}"
    # "${DATA_LOAD_ARGS[*]}"
)
ARGS="$(join_by ' ' ${gpt_args[*]})"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "ARGS: ${ARGS}"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
# gpt_args+="${ds_args}"
# gpt_args+="${DATA_LOAD_ARGS}"
