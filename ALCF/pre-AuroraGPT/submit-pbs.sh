#!/bin/bash --login
#

cd "${PBS_O_WORKDIR}" || exit

# cd "${PBS_O_WORKDIR}"
#
# echo "PBS_O_WORKDIR: ${PBS_O_WORKDIR}"
#
# echo "__________________________________________________________________________________"
# cd ~/datascience/foremans/locations/polaris/projects/saforem2/Megatron-DS-Benchmarking/
# echo "pwd: $(pwd)"
# echo "__________________________________________________________________________________"

# SOURCE=${BASH_SOURCE[0]}
# while [ -L "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
#   DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
#   SOURCE=$(readlink "$SOURCE")
#   [[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
# done
# DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"

# HERE=$(python3 -c 'import os; print(os.getcwd())')
# ALCF_DIR="${HERE}/ALCF"

ALCF_DIR="$(dirname $(dirname $(python3 -c 'import megatron; print(megatron.__file__)' | tail -1)))/ALCF"
PARENT=$(dirname "${ALCF_DIR}")
echo "+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+"
echo "ALCF_DIR: ${ALCF_DIR}"
echo "PARENT: ${PARENT}"
echo "+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+"


TSTAMP=$(tstamp)
echo "┌──────────────────────────────────────────────────────────────────┐"
#####"│ Job Started at 2023-08-04-121535 on polaris-login-04 by foremans │"
echo "│ Job Started at ${TSTAMP} on $(hostname) by $USER │"
echo "│ in: ${PARENT}"
echo "└──────────────────────────────────────────────────────────────────┘"
# echo "------------------------------------------------------------------------"

getValFromFile() {
  FILE=$1
  KEY=$2
  echo "getting ${KEY} from ${FILE}"
  if [[ -f "${FILE}" ]]; then
    VAL="$(cat "${FILE}" | grep -E "^${KEY}=" | sed "s/${KEY}=//g" | sed 's/\"//g')"
    echo "setting ${KEY}: ${VAL}"
    export "${KEY}"="${VAL}"
  fi
}

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
#
# sourceFile "${DIR}/setup.sh"
# sourceFile "${DIR}/model.sh"
# sourceFile "${DIR}/args.sh"
# sourceFile "${DIR}/launch.sh"
#
# export USE_ACTIVATION_CHECKPOINTING=1  # 1 | 0
# export SEQ_LEN=${SEQ_LEN:-1024}
# export MPSIZE=${MPSIZE:-1}
# export PPSIZE=${PPSIZE:-1}
# export SPSIZE=${SPSIZE:-1}
# export MICRO_BATCH=${MICRO_BATCH:-1}
# export ZERO_STAGE=${ZERO_STAGE:-1}  # 0 | 1 | 2 | 3
# export NHOSTS="$NHOSTS"
# export GRADIENT_ACCUMULATION_STEPS=${GAS:-1}
# export USE_SEQUENCE_PARALLEL=${USE_SEQUENCE_PARALLEL:-0}  # 1 | 0
#
#
# export MODEL_SIZE_KEY="GPT1_5B"
# export SEQ_LEN=1024
# export USE_FLASH_ATTN=1
# export MICRO_BATCH=4
# export WORLD_SIZE=8
# export SP_TYPE="ds"
# export SPSIZE=8
# export PPSIZE=1
# export MPSIZE=1
# export ZERO_STAGE=3
# export USE_SEQUENCE_PARALLEL=0


# getValFromFile "${DIR}/model.sh" MODEL_SIZE
# getValFromFile "${DIR}/args.sh" PPSIZE
# getValFromFile "${DIR}/args.sh" MPSIZE
# getValFromFile "${DIR}/args.sh" MICRO_BATCH
# getValFromFile "${DIR}/args.sh" GRADIENT_ACCUMULATION_STEPS
#
# MODEL_SIZE="${MODEL_SIZE}"
# PPSIZE="${PPSIZE}"
# MPSIZE="${MPSIZE}"
# MICRO_BATCH="${MICRO_BATCH}"
# GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS}"

QUEUE=$1
NUM_NODES=$2
DURATION=$3
PROJECT=$4

# MODEL_SIZE_KEY=$5
# SEQ_LEN=$6
# USE_FLASH_ATTN=$7
# MICRO_BATCH=$8
# GAS=$9
# SP_TYPE=$10

# MODEL_SIZE_KEY="GPT6_7B" SEQ_LEN=2048 USE_FLASH_ATTN=0 MICRO_BATCH=1 GAS=1 SP_TYPE="deepspeed" ./ALCF/submit-pbs.sh debug-scaling 4 00:30:00 datascience

# export MICRO_BATCH=${MICRO_BATCH:-1}
# export MICRO_BATCH="${MICRO_BATCH}"
# export MODEL_SIZE="${MODEL_SIZE}"
# # export GAS="${GRADIENT_ACCUMULATION_STEPS}"
# export GRADIENT_ACCUMULATION_STEPS=${GAS:-1}
#
# export DDP_IMPL="local"   # FSDP | local | torch
# # export USE_FLASH_ATTN=${USE_FLASH_ATTN:-0}  # 1 | 0
# # export USE_ACTIVATION_CHECKPOINTING=1  # 1 | 0
# export SEQ_LEN=${SEQ_LEN:-1024}
# # export MPSIZE=${MPSIZE:-1}
# export PPSIZE=${PPSIZE:-1}
# export SPSIZE=${SPSIZE:-1}
# export MICRO_BATCH=${MICRO_BATCH:-1}
# export ZERO_STAGE=${ZERO_STAGE:-1}  # 0 | 1 | 2 | 3
# # export NHOSTS="$NHOSTS"
# export GRADIENT_ACCUMULATION_STEPS=${GAS:-1}
# export USE_SEQUENCE_PARALLEL=${USE_SEQUENCE_PARALLEL:-0}  # 1 | 0
#

if [ -z "${MODEL_SIZE_KEY}" ]; then
  echo "ERROR: MODEL_SIZE_KEY not set"
  exit 1
fi

if [ -z "${SEQ_LEN}" ]; then
  echo "ERROR: SEQ_LEN not set"
  echo "Using default SEQ_LEN=2048"
  echo "Set SEQ_LEN=XXXX to change"
  SEQ_LEN=2048
fi

if [ -z "${USE_FLASH_ATTN}" ]; then
  echo "ERROR: USE_FLASH_ATTN not set"
  echo "Not using flash attn! Set USE_FLASH_ATTN=1 to use"
  USE_FLASH_ATTN=0
fi

if [ -z "${MICRO_BATCH}" ]; then
  echo "ERROR: MICRO_BATCH not set"
  echo "Using MICRO_BATCH=1"
  MICRO_BATCH=1
fi

if [ -z "${GAS}" ]; then
  echo "ERROR: GAS not set"
  echo "Using GAS=1"
  GAS=1
fi

if [ -z "${SP_TYPE}" ]; then
  echo "ERROR: SP_TYPE not set"
  echo "Using SP_TYPE=megatron"
  SP_TYPE="megatron"
fi

export GAS="${GAS}"
export SEQ_LEN="${SEQ_LEN}"
export SP_TYPE="${SP_TYPE}"
export MICRO_BATCH="${MICRO_BATCH}"
export MODEL_SIZE_KEY="${MODEL_SIZE_KEY}"
export USE_FLASH_ATTN="${USE_FLASH_ATTN}"

echo "-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-"
echo "| MODEL_SIZE_KEY: ${MODEL_SIZE_KEY}"
echo "| SEQ_LEN: ${SEQ_LEN}"
echo "| USE_FLASH_ATTN: ${USE_FLASH_ATTN}"
echo "| MICRO_BATCH: ${MICRO_BATCH}"
echo "| GAS: ${GAS}"
echo "| SP_TYPE: ${SP_TYPE}"
echo "-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-"

export QUEUE="${QUEUE}"
export DURATION="${DURATION}"
export TSTAMP="${TSTAMP}"
export NUM_NODES="${NUM_NODES}"
export PROJECT="${PROJECT}"

RUN_NAME="N${NUM_NODES}-${TSTAMP}"
# RUN_NAME="mb${MICRO_BATCH}-gas${GAS}-${RUN_NAME}"
# RUN_NAME="GPT3-${MODEL_SIZE}-${RUN_NAME}"
RUN_NAME="${MODEL_SIZE_KEY}-${SP_TYPE}-mb${MICRO_BATCH}-gas${GAS}-seqlen${SEQ_LEN}-${RUN_NAME}"
export RUN_NAME="${RUN_NAME}"

echo "QUEUE=$QUEUE"
echo "PROJECT=$PROJECT"
echo "DURATION=$DURATION"
echo "TSTAMP=$TSTAMP"
echo "NUM_NODES=$NUM_NODES"
echo "RUN_NAME: ${RUN_NAME}"
# echo "MODEL_SIZE=$MODEL_SIZE"
# echo "GAS=$GRADIENT_ACCUMULATION_STEPS"

# QSUB_ARGS=(
#   "-q ${QUEUE}"
#   "-A ${PROJECT}"
#   "-N ${RUN_NAME}"
#   "-l select=${NUM_NODES}"
#   "-l walltime=${DURATION}"
#   "-l filesystems=eagle:home:grand"
#   "${DIR}/submit.sh"
# )

OUTPUT=$(qsub \
  -q "${QUEUE}" \
  -A "${PROJECT}" \
  -N "${RUN_NAME}" \
  -l select="${NUM_NODES}" \
  -l walltime="${DURATION}" \
  -l filesystems=eagle:home:grand \
  "${ALCF_DIR}/submit.sh")

# OUTPUT=$(qsub "${QSUB_ARGS[@]}")

PBS_JOBID=$(echo "${OUTPUT}" | cut --delimiter="." --fields=1)
export PBS_JOBID="${PBS_JOBID}"
# echo "${TSTAMP} ${PBS_JOBID} "

PBS_JOBSTR=(
  "PBS_JOBID=${PBS_JOBID}"
  "QUEUE=$QUEUE"
  "PROJECT=$PROJECT"
  "DURATION=$DURATION"
  "TSTAMP=$TSTAMP"
  "NUM_NODES=$NUM_NODES"
  # "MODEL_SIZE=$MODEL_SIZE"
  "RUN_NAME: ${RUN_NAME}"
)
  # "GAS=$GRADIENT_ACCUMULATION_STEPS"

TODAY=$(echo "${TSTAMP}" | cut --delimiter="-" --fields=1,2,3)
OUTFILE="${PARENT}/pbslogs/${TODAY}/${PBS_JOBID}.txt"

if [[ ! -d $(dirname "${OUTFILE}") ]]; then
  mkdir -p "$(dirname "${OUTFILE}")"
fi

echo "Writing PBS_JOBSTR to ${OUTFILE}"
echo "${PBS_JOBSTR[@]}" >> "${OUTFILE}"
# echo "${PBS_JOBSTR[@]}" | tee -a "${OUTFILE}"

echo "┌───────────────────────────────────────────┐"
echo "│ To view job output, run: \`pbstail ${PBS_JOBID}\` │"
echo "└───────────────────────────────────────────┘"
