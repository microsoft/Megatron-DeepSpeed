#!/bin/bash --login

function FindMegatron() {
    MEGATRON_INSTALL=$(python3 -c 'import megatron; print(megatron.__file__)' | tail -1)
    MEGATRON_DIR=$(dirname $(dirname $(python3 -c 'import megatron; print(megatron.__file__)' | tail -1)))
}

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

WORLD_SIZE="${NGPUS}"
PARALLEL_SIZE="${WORLD_SIZE}"
echo "NHOSTS * (NGPU / HOST) = $NHOSTS * $NGPU_PER_HOST = $NGPUS"

export MODEL_SIZE_KEY="${MODEL_SIZE_KEY:-GPT13B}"
echo "==========================+"
echo "Using ${MODEL_SIZE_KEY}"
echo "==========================+"

sourceFile "${ALCF_DIR}/model.sh"

MODEL_TYPE=${MODEL_TYPE:-gpt}

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ Model Parallel / Pipeline Parallel ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
export DDP_IMPL="local"   # FSDP | local | torch
export USE_ACTIVATION_CHECKPOINTING=1  # 1 | 0
export SEQ_LEN=${SEQ_LEN:-2048}
export PPSIZE=${PPSIZE:-1}
export MICRO_BATCH=${MICRO_BATCH:-1}
export GRADIENT_ACCUMULATION_STEPS=${GAS:-1}
export MODEL_TYPE=${MODEL_TYPE:-"gpt"} # set bert or gpt
export SP_TYPE=${SP_TYPE:-"megatron"} # set ds or megatron
export ZERO_STAGE=${ZERO_STAGE:-1}
export MPSIZE=${MPSIZE:-${WORLD_SIZE:-1}}
export SPSIZE=${SPSIZE:-1}

#
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

GLOBAL_BATCH=$(( NGPUS * MICRO_BATCH * GRADIENT_ACCUMULATION_STEPS ))
echo "GB = NGPUS * MB * GAS = ${NGPUS} * ${MICRO_BATCH} * ${GRADIENT_ACCUMULATION_STEPS} = ${GLOBAL_BATCH}"

GLOBAL_BATCH=$(( GLOBAL_BATCH / MPSIZE / PPSIZE / SPSIZE))
echo "GB = (NGPUS * MB * GAS) / (MP * PP * SP) = (${NGPUS} * ${MICRO_BATCH} * ${GRADIENT_ACCUMULATION_STEPS}) / (${MPSIZE} * ${PPSIZE} * ${SPSIZE}) = ${GLOBAL_BATCH}"

if [[ "${GLOBAL_BATCH}" == 0 ]]; then
    GLOBAL_BATCH=1
fi
# [ "${GLOBAL_BATCH:-${GLOBAL_BATCH}}" == 0 ] && GLOBAL_BATCH=1 || echo "GLOBAL_BATCH: ${GLOBAL_BATCH}"
export GLOBAL_BATCH="$GLOBAL_BATCH"

DPSIZE=$(( $WORLD_SIZE / $PPSIZE / $MPSIZE ))

# echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
# echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"

echo "--------------------------------"
echo "GLOBAL_BATCH=${GLOBAL_BATCH}"
echo "USING DPSIZE: ${DPSIZE}"
echo "--------------------------------"

# ┏━━━━━━━━━━━━┓
# ┃ Data paths ┃
# ┗━━━━━━━━━━━━┛
if [[ $(hostname) == nid* || $(hostname) == login* ]]; then
    DATA_PARENT="/global/homes/f/foremans/m3957/foremans/projects/saforem2/Megatron-DeepSpeed"
    DATA_TYPE="BookCorpusDataset_text_document"
elif [[ $(hostname) == theta* || $(hostname) == x3* ]]; then
    DATA_PARENT="/lus/eagle/projects/datasets/Megatron-DeepSpeed/GenSLMSubSample200k"  # GenSLMSubSample200k"
    # DATA_TYPE="GenSLMSubSample200k"
    # DATA_PARENT="/home/foremans/datascience/foremans/locations/polaris/projects/argonne-lcf/Megatron-DeepSpeed"
    # DATA_TYPE="books-0001_text_document"
    # DATA_PARENT="/lus/grand/projects/fallwkshp23/datasets/GenSLMSubSample200k"
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
# #
# [ "$(hostname)==login*" ] && DATA_PARENT="/global/homes/f/foremans/m3957/foremans/projects/saforem2/Megatron-DeepSpeed"_
# [ "$(hostname)==nid*" ] && DATA_PARENT="/global/homes/f/foremans/m3957/foremans/projects/saforem2/Megatron-DeepSpeed"
# # [ "$(hostname)==theta*" ] && DATA_PARENT="/lus/eagle/projects/datasets/BookCorpusDataset"
# [ "$(hostname)==theta*" ] && DATA_PARENT="/lus/grand/projects/fallwkshp23/datasets/GenSLMSubSample200k"
# [ "$(hostname)==x3*" ] && DATA_PARENT="/lus/grand/projects/fallwkshp23/datasets/GenSLMSubSample200k"
# # "/lus/grand/projects/datascience/foremans/locations/thetaGPU/projects/saforem2/Megatron-DS-Benchmarking"
# # /lus/grand/projects/fallwkshp23/datasets/GenSLMSubSample200k/dataset/genslm_subsample_200k_sequence_document"
# # [ "$(hostname)==x3*" ] && DATA_PARENT="/lus/eagle/projects/datasets/BookCorpusDataset"
# # /lus/grand/projects/datascience/foremans/locations/thetaGPU/projects/saforem2/Megatron-DS-Benchmarking"
# # /lus/grand/projects/fallwkshp23/datasets/GenSLMSubSample200k/dataset/genslm_subsample_200k_sequence_document.bin
#
# # DATA_PATH=/lus/grand/projects/datascience/vsastry/genslm_subsample_200k_sequence_document/genslm_subsample_200k_sequence_document
# DATA_DIR="${DATA_PARENT}/dataset"
# DATA_PATH="${DATA_DIR}/genslm_subsample_200k_sequence_document"
# VOCAB_FILE="${DATA_DIR}/gpt2-vocab.json"
# MERGE_FILE="${DATA_DIR}/gpt2-merges.txt"

# ┏━━━━━━━━━━━━━━━━━━━┓
# ┃ FILE I/O SETTINGS ┃
# ┗━━━━━━━━━━━━━━━━━━━┛
RUN_STR="gb${GLOBAL_BATCH}_mb${MICRO_BATCH}"
RUN_STR="nl${NLAYERS}_hs${HIDDEN}_${RUN_STR}"
RUN_STR="mp${MPSIZE}_pp${PPSIZE}_sp${SPSIZE}_${RUN_STR}"
RUN_STR="z${ZERO_STAGE}_seqlen${SEQ_LEN}_${RUN_STR}"
RUN_STR="${MODEL_SIZE}_${RUN_STR}"

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

RUN_STR="${MODEL_TYPE}_${RUN_STR}"

OUTPUT_DIR="${PARENT}/outputs/${RUN_STR}"
CHECKPOINT_DIR="${PARENT}/checkpoints/$RUN_STR"
TENSORBOARD_DIR="${PARENT}/outputs/${RUN_STR}/tensorboard"

DATE=$(date)
export DATE="${DATE}"
export RUN_STR="${RUN_STR}"
export MODEL_SIZE="$MODEL_SIZE"
export TENSORBOARD_DIR=$TENSORBOARD_DIR
export OUTPUT_DIR=$OUTPUT_DIR
mkdir -p "$OUTPUT_DIR/tensorboard/wandb"
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$TENSORBOARD_DIR"
mkdir -p "$OUTPUT_DIR"
echo "OUTPUT TO: ${OUTPUT_DIR}"

# if [[ -z "${NVME_PATH}" ]]; then
#   echo "NVME_PATH: $NVME_PATH"
# else
#   if [[ $(hostname) == x* ]]; then
#     export NVME_PATH="/local/scratch/"
#   elif [[ $(hostname) == theta* ]]; then
#     export NVME_PATH="/raid/scratch/"
#   else
#     export NVME_PATH="/tmp/"
#   fi
# fi

# echo "NVME_PATH: ${NVME_PATH}"

if [[ $MODEL_TYPE == "gpt" ]] ; then
    DATA_LOAD_ARGS="--data-path $DATA_PATH --vocab-file $VOCAB_FILE --merge-file $MERGE_FILE"
else
    DATA_LOAD_ARGS=""
fi

# Set to cpu for offloading to cpu for larger models
OFFLOAD_DEVICE="${OFFLOAD_DEVICE:-cpu}"
CPU_OPTIM=" --cpu-optimizer"

# # Set to none and empty string for no cpu offloading
# OFFLOAD_DEVICE="none"  
# CPU_OPTIM=" "


[ "${WANDB_MODE}" == "disabled" ] && WANDB_ENABLE="false" || WANDB_ENABLE="true"
echo "WANDB_ENABLE: ${WANDB_ENABLE}"

# ┏━━━━━━━━━━━━━━━━━━┓
# ┃ DeepSpeed Config ┃
# ┗━━━━━━━━━━━━━━━━━━┛
DS_CONFIG=${PARENT}/ds_config-gpt.json
echo "!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!"
echo " DS_CONFIG: ${DS_CONFIG}"
echo "!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!"

if [[ $ZERO_STAGE == "3" ]] ; then
    cat <<EOT > "$DS_CONFIG"
    {
        "train_micro_batch_size_per_gpu": $MICRO_BATCH,
        "steps_per_print": 1,
        "wall_clock_breakdown" : true,
        "gradient_accumulation_steps": $GRADIENT_ACCUMULATION_STEPS,
        "gradient_clipping": 1.0,
        "zero_optimization": {
        "stage": 3,
        "stage3_max_live_parameters": 3e9,
        "stage3_max_reuse_distance": 3e9,
        "stage3_param_persistence_threshold": 1e5,
        "stage3_prefetch_bucket_size": 1e9,
        "contiguous_gradients": true,
        "overlap_comm": true,
        "reduce_bucket_size": 90000000,
        "sub_group_size": 5e7,
        "offload_param": {
        "device": "cpu",
        "pin_memory": true
    },
    "offload_optimizer": {
    "device": "cpu",
    "buffer_count": 4,
    "pipeline_read": false,
    "pipeline_write": false,
    "pin_memory": true
}
},
"fp16": {
"enabled": true,
"initial_scale_power" : 12,
"loss_scale_window": 1000,
"hysteresis": 2,
"min_loss_scale": 1
},
"aio": {
"block_size": 1048576,
"queue_depth": 16,
"single_submit": false,
"overlap_events": true,
"thread_count": 2
},
"flops_profiler": {
"enabled": true,
"profile_step": 1,
"module_depth": -1,
"top_modules": 3,
"detailed": true,
"output_file": null
},
"comms_logger": {
"enabled": true,
"verbose": false,
"prof_all": false,
"debug": false
},
"wandb": {
"enabled": $WANDB_ENABLE,
"project": "GenSLM-Megatron-DS"
}
}
EOT
else
    cat <<EOT > "$DS_CONFIG"
    {
        "train_micro_batch_size_per_gpu": $MICRO_BATCH,
        "gradient_accumulation_steps": $GRADIENT_ACCUMULATION_STEPS,
        "steps_per_print": 1,
        "wall_clock_breakdown" : true,
        "zero_force_ds_cpu_optimizer": false,
        "zero_optimization": {
        "stage": $ZERO_STAGE,
        "allgather_partitions": true,
        "reduce_scatter": true,
        "allgather_bucket_size": 5e8,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "offload_param": {
        "device": "cpu",
        "nvme_path": "/raid/scratch",
        "pin_memory": false
    },
    "offload_optimizer": {
    "device": "cpu",
    "nvme_path": "/raid/scratch/"
}
},
"scheduler": {
"type": "WarmupLR",
"params": {
"warmup_min_lr": 0,
"warmup_max_lr": 0.001,
"warmup_num_steps": 1000
}
},
"fp16": {
"enabled": true,
"initial_scale_power": 12
},
"flops_profiler": {
"enabled": true,
"profile_step": 1,
"module_depth": -1,
"top_modules": 3,
"detailed": true,
"output_file": null
},
"comms_logger": {
"enabled": true,
"verbose": false,
"prof_all": false,
"debug": false
},
"wandb": {
"enabled": $WANDB_ENABLE,
"project": "GenSLM-Megatron-DS"
}
}
EOT
fi

# ┏━━━━━━━━━━━━━━━━━━━━━┓
# ┃ DeepSpeed Arguments ┃
# ┗━━━━━━━━━━━━━━━━━━━━━┛
if [[ "$DDP_IMPL" != "FSDP" ]] ; then
    ds_args=""
    ds_args=" --deepspeed ${ds_args}"
    ds_args=" --deepspeed_mpi ${ds_args}"
    ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
    ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"
    if [[ "$PPSIZE" == 1 ]]; then
        ds_args="--no-pipeline-parallel ${ds_args}"
    else
        ds_args=" --pipeline-model-parallel-size ${PPSIZE} ${ds_args}"
    fi
    if [[ "$USE_ACTIVATION_CHECKPOINTING" == 1 ]]; then
        ds_args=" --deepspeed-activation-checkpointing ${ds_args}"
    fi
fi

# ┏━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ MEGATRON-LM SETTINGS ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━┛
gpt_args=(
    "--no-async-tensor-model-parallel-allreduce"
    "--seed ${RANDOM}"
    "--DDP-impl ${DDP_IMPL}"
    "--pipeline-model-parallel-size ${PPSIZE}"
    "--tensor-model-parallel-size ${MPSIZE}"
    "--ds-sequence-parallel-size ${SPSIZE}"
    "--num-layers ${NLAYERS}"
    "--hidden-size ${HIDDEN}"
    "--num-attention-heads ${ATEN_HEADS}"
    "--micro-batch-size ${MICRO_BATCH}"
    "--global-batch-size ${GLOBAL_BATCH}"
    "--seq-length ${SEQ_LEN}"
    "--max-position-embeddings ${SEQ_LEN}"
    "--train-iters 10"
    "--lr-decay-iters 320000"
    "--num-workers 0"
    "$DATA_LOAD_ARGS"
    "--data-impl mmap"
    "--split 949,50,1"
    "--distributed-backend nccl"
    "--lr 0.00015"
    "--lr-decay-style cosine"
    "--min-lr 1.0e-5"
    "--weight-decay 1e-2"
    "--clip-grad 1.0"
    "--lr-warmup-fraction .01"
    "--log-interval 1"
    "--save-interval 1000"
    "--eval-interval 1000"
    "--eval-iters 0"
    "--override-opt_param-scheduler"
    "--tensorboard-dir ${TENSORBOARD_DIR}"
    "--log-timers-to-tensorboard"
    "--tensorboard-log-interval 1"
)


if [[ "$USE_ACTIVATION_CHECKPOINTING" == 1 ]]; then
    gpt_args+=(
    "--checkpoint-activations"
    "--checkpoint-num-layers 1"
)
fi

if [[ "$DDP_IMPL" != "FSDP" ]] ; then
    gpt_args+=(
    # "${gpt_args[*]}"
    "--fp16"
)
else
    gpt_args+=(
    "--bf16"
)
fi

# Flash Attention v1
if [[ "${USE_FLASH_ATTN1}" || "${USE_FLASH_ATTN_V1}" ]]; then
    [ "${USE_FLASH_ATTN}" ] && gpt_args+=("--use-flash-attn-v1")
    [ "${USE_FLASH_ATTN1}" ] && gpt_args+=("--use-flash-attn-v1")
    [ "${USE_FLASH_ATTN_V1}" ] && gpt_args+=("--use-flash-attn-v1")
# Flash Attention 2
elif [[ "${USE_FLASH_ATTN2}" || "${USE_FLASH_ATTN_V2}" ]]; then
    [ "${USE_FLASH_ATTN2}" ] && gpt_args+=("--use-flash-attn-v2")
    [ "${USE_FLASH_ATTN_V2}" ] && gpt_args+=("--use-flash-attn-v2")
# Triton + Flash Attn
elif [[ "${USE_FLASH_ATTN_TRITON}" ]]; then
    [ "${USE_FLASH_ATTN_TRITON}" ] && gpt_args+=("--use-flash-attn-triton")
fi

if [[ "$USE_SEQUENCE_PARALLEL" == 1 ]]; then
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    gpt_args+=(
    "--sequence-parallel"
)
fi

if [[ "$ZERO_STAGE" > "0" ]] ; then
    gpt_args+=(
    "--cpu-optimizer"
)
fi

export gpt_args=(
"${gpt_args[*]}"
"${ds_args[*]}"
)
ARGS="$(join_by ' ' ${gpt_args[*]})"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "ARGS: ${ARGS}"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
