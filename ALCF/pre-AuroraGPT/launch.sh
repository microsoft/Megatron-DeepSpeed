#!/bin/bash --login

HOST=$(hostname)

function WhereAmI() {
    python3 -c 'import os; print(os.getcwd())'
}

# function join_by {
#     local d=${1-} f=${2-}; if shift 2; then printf %s "$f" "${@/#/$d}"; fi; 
# }

USER=$(whoami)
HERE=$(WhereAmI)
ALCF_DIR="${HERE}/ALCF"
PARENT=$(dirname "${ALCF_DIR}")

echo "+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+"
echo "ALCF_DIR: ${ALCF_DIR}"
echo "PARENT: ${PARENT}"
echo "+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+"


function sourceFile() {
    FILE="$1"
    echo "source-ing ${FILE}"
    if [[ -f "${FILE}" ]]; then
        # shellcheck source=./setup.sh
        source "${FILE}"
    else
        echo "ERROR: UNABLE TO SOURCE ${FILE}"
    fi
}

MASTER_ADDR=$(uname -n)
MASTER_PORT=20010
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
MPI_WRAPPER="${SCRIPT_DIR}/mpi_wrapper"

# sourceFile "${ALCF_DIR}/args.sh"

# MAIN="${PARENT}/pretrain_${MODEL_TYPE}.py"
MAIN="${PARENT}/pretrain_gpt_alcf.py"

printJobInfo() {
    echo "Job started at: ${TSTAMP} on $(hostname)"
    echo "Job running in: ${DIR}"
    echo "Training Llama2 with ${MODEL_SIZE} parameters"
    echo "Writing logs to: ${OUTPUT_DIR}"
    echo 'to view output: tail -f $(tail -1 logfiles)'
    echo "i.e. tail -f $(tail -1 "${PARENT}"/logfiles)"
}

launchJob() {
    echo "using: $(which python3)" | tee -a "${OUTPUT_LOG}"
    printJobInfo | tee -a "${OUTPUT_LOG}"
    echo EXEC="${EXEC}" | tee -a "${OUTPUT_LOG}"
    echo "Writing logs to: ${OUTPUT_LOG}" | tee -a "${OUTPUT_LOG}"
    # ARGS="$@"
    # export ARGS="$ARGS"
    ${EXEC} "$@" # >> "${OUTPUT_LOG}" 2>&1 &
}

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ Use all available GPUs a single nodes ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
fullNode() {
    echo "fullNode started"
    echo "MPI_COMMAND ${MPI_COMMAND}"
    echo "MPI_DEFAULTS ${MPI_DEFAULTS}"
    echo "NGPUS ${NGPUS}"
    echo "hostfile ${DIR}/hostfile"
    echo "MAIN ${MAIN}"
    echo "gpt_args ${ARGS}"
    NHOSTS=$(wc -l < "${HOSTFILE}")
    NGPU_PER_HOST=$(nvidia-smi -L | wc -l)
    NGPUS=$((${NHOSTS}*${NGPU_PER_HOST}))
    # hostname > $DIR/hostfile
    echo "\
        Running on $NHOSTS hosts \
        with $NGPU_PER_HOST GPUs each \
        for a total of $NGPUS GPUs"
    _EXEC=(
        "${MPI_COMMAND}"
        "${MPI_DEFAULTS}"
        "${MPI_ELASTIC}"
        "${MPI_WRAPPER}"
        "${MASTER_ADDR}"
        "${MASTER_PORT}"
        "${MAIN}"
        "${ARGS}"
        # "${ds_args}"
    )
    # EXEC=$(join_by ' ' "${EXEC[*]}")
    EXEC="${EXEC[*]}"
    OUTPUT_LOG="${OUTPUT_DIR}/logs/$USER-$HOST-nhosts${NHOSTS}-ngpu${NGPUS}-$TSTAMP.log"
    mkdir -p "$(dirname "${OUTPUT_LOG}")"
    echo "${OUTPUT_LOG}" >> "${PARENT}/logfiles"
    printJobInfo | tee -a "${OUTPUT_LOG}"
    launchJob "$@" 2>&1 | tee "${OUTPUT_LOG}"
}


function setupSrunOld() {
    if [[ $(hostname) == login* || $(hostname) == nid* ]]; then
        export NODELIST="${SLURM_JOB_NODELIST:-$(hostname)}"
        export MACHINE="Perlmutter"
        export NHOSTS="${SLURM_NNODES:-1}"
        export NGPU_PER_HOST="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}"
        export NGPUS="$(( NHOSTS * NGPU_PER_HOST ))"
        export SRUN_EXEC="srun -N ${NHOSTS} -n ${NGPUS} -l -u"
    else
        echo "Skipping setupSrun() on $(hostname)"
    fi
}

function setupSrun() {
    if [[ $(hostname) == login* || $(hostname) == nid* ]]; then
        export NHOSTS="${SLURM_NNODES:-1}"
        export NGPU_PER_HOST="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}"
        export NGPUS="$(( NHOSTS * NGPU_PER_HOST ))"
        export SRUN_EXEC="srun --gpus ${NGPUS} --gpus-per-node ${NGPU_PER_HOST} -N ${NHOSTS} -n ${NGPUS} -l -u --verbose"
    else
        echo "Skipping setupSrun() on $(hostname)"
    fi
}


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ Use all available GPUs on all available nodes ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
elasticDistributed() {
    if [[ $(hostname) == theta* || $(hostname) == x3* ]]; then
        if [[ $(hostname) == theta* ]]; then
            echo "Setting up ThetaGPU from $(hostname)"
            HOSTFILE="${HOSTFILE:-${COBALT_NODEFILE}}"
        elif [[ $(hostname) == x3* ]]; then
            echo "Setting up Polaris from $(hostname)"
            HOSTFILE="${HOSFILE:-${PBS_NODEFILE}}"
        else
            echo "Unknown hostname $(hostname)"
            exit 1
        fi
        NHOSTS=$(wc -l < "${HOSTFILE}")
        NGPU_PER_HOST=$(nvidia-smi -L | wc -l)
        NGPUS="$(( NHOSTS * NGPU_PER_HOST ))"
        export MASTER_ADDR="127.0.0.1"
        export MASTER_PORT="5432"
        EXEC_STR=(
            "${MPI_COMMAND}"
            "${MPI_DEFAULTS}"
            "${MPI_ELASTIC}"
            "$(which python3)"
            "${MAIN}"
            "${ARGS}"
            # "${gpt_args}"
            # "${ds_args}"
        )
    elif [[ $(hostname) == nid*  || $(hostname) == login* ]]; then
        echo "Setting up from Perlmutter on $(hostname)"
        # NHOSTS=${SLURM_NNODES-1}
        MACHINE="Perlmutter"
        setupPerlmutter
        setupSrun
        echo "SRUN_EXEC: ${SRUN_EXEC}"
        export MASTER_ADDR="$SLURMD_NODENAME"
        EXEC_STR=(
            "${SRUN_EXEC}"
            "$(which python3)"
            "${MAIN}"
            "${ARGS}"
            # "${gpt_args}"
            # "${ds_args}"
        )
    else
        echo "Unexpected hostname $(hostname)"
    fi
    export WORLD_SIZE="${NGPUS}"
    echo "\
        Running on ${NHOSTS} hosts \
        with ${NGPU_PER_HOST} GPUs each \
        for a total of ${NGPUS} GPUs"
    EXEC="${EXEC_STR[*]}"
    OUTPUT_LOG="${OUTPUT_DIR}/logs/$USER-$HOST-nhosts${NHOSTS}-ngpu${NGPUS}-$TSTAMP.log"
    echo "EXEC_STR: ${EXEC_STR}"
    echo "Writing logs to: ${OUTPUT_LOG}"
    mkdir -p "$(dirname "${OUTPUT_LOG}")"
    echo "${OUTPUT_LOG}" >> "${PARENT}/logfiles"
    printJobInfo | tee -a "${OUTPUT_LOG}"
    launchJob "$@" >> "${OUTPUT_LOG}" 2>&1 &
    PID=$!
    wait $PID
}
