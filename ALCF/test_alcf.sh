#!/bin/bash --login
#
# Run complete test of
# https://github.com/argonne-lcf/Megatron-DeepSpeed
# on {Polaris, Sunspot, Sirius} @ ALCF
# to launch (inside an interactive `qsub -I` job) on Polaris:
#
# ```bash`
# $ git clone https://github.com/argonne-lcf/Megatron-DeepSpeed
# $ cd Megatron-DeepSpeed/ALCF
# $ bash test_alcf.sh
# ````

# EXIT ON ERROR(s)
set -euxo pipefail

NOW="$(date "+%Y-%m-%d-%H%M%S")"

setup_conda_sunspot() {
    if [[ -z "${CONDA_PREFIX-}" && -z "${VIRTUAL_ENV-}" ]]; then
        shell_name=$(echo "${SHELL}" | tr "\/" "\t" | awk '{print $NF}')
        eval "$(~/miniconda3/bin/conda shell hook -s posix)"
        conda activate q4-drop
    else
        echo "Found existing python at: $(which python3)"
    fi
}

setup_conda_sirius() {
    if [[ -z "${CONDA_PREFIX-}" && -z "${VIRTUAL_ENV-}" ]]; then
        export MAMBA_ROOT_PREFIX=/lus/tegu/projects/PolarisAT/foremans/micromamba
        shell_name=$(echo "${SHELL}" | tr "\/" "\t" | awk '{print $NF}')
        eval "$("${MAMBA_ROOT_PREFIX}/bin/micromamba" shell hook --shell ${shell_name})"
        micromamba activate 2024-04-23
    else
        echo "Found existing python at: $(which python3)"
    fi
}

setup_conda_polaris() {
    if [[ -z "${CONDA_PREFIX-}" && -z "${VIRTUAL_ENV-}" ]]; then
        # export CUDA_HOME=/soft/compilers/cudatoolkit/cuda-12.2.2
        # && export MAMBA_ROOT_PREFIX=/eagle/argonne_tpc/micromamba && eval "$("${MAMBA_ROOT_PREFIX}/bin/micromamba" shell hook -s posix)" ; mm activate 2024-04-25
        export MAMBA_ROOT_PREFIX=/eagle/argonne_tpc/micromamba
        shell_name=$(echo "${SHELL}" | tr "\/" "\t" | awk '{print $NF}')
        eval "$("${MAMBA_ROOT_PREFIX}/bin/micromamba" shell hook -s posix)"
        micromamba activate 2024-04-25
    else
        echo "Found existing python at: $(which python3)"
    fi
}


function setEnv() {
    local virtual_env="${VIRTUAL_ENV-}"
    local conda_prefix="${CONDA_PREFIX-}"
    if [[ -n "${conda_prefix}" && -z "${virtual_env}" ]]; then
        echo "Using conda from: ${conda_prefix}"
    elif [[ -n "${virtual_env}" && -z "${conda_prefix}" ]]; then
        echo "Using virtual_env from: ${virtual_env}"
    elif [[ -n "${virtual_env}" && -n "${conda_prefix}" ]]; then
        echo "Using virtual_env: ${virtual_env} on top of CONDA: ${conda_prefix}"
    elif [[ -z "${conda_prefix}" && -z "${virtual_env}" ]]; then
        echo "No conda_prefix or virtual_env found in environment..."
        echo "Setting up conda"
        # setup_conda
        # ---- [SunSpot] ------- || ---- [Aurora] --------------
        if [[ $(hostname) == x1* || $(hostname) == x4* ]]; then
            source "${WORKING_DIR}/ALCF/sunspot-env.sh" || exit
            # ----- [Aurora] -----------------------------------
            if [[ -z "${conda_prefix}" && -z "${virtual_env}" ]]; then
                if [[ $(hostname) == x4* ]]; then
                    eval "$(conda shell.zsh hook)" && conda activate anl_release_q4v2
                # ----- [SunSpot] ----------------------------------
                elif [[ $(hostname) == x1* ]]; then
                    echo "Running on SunSpot !!"
                    setup_conda_sunspot
                    # eval "$(/home/foremans/miniconda3/bin/conda shell.zsh hook)" && conda activate q4-drop
                fi
            fi
        # ----- [Polaris] ---------------------------------------
        elif [[ $(hostname) == x3* ]]; then
            if [[ "${PBS_O_HOST}" == sirius* ]]; then
                echo "Running on Sirius !!"
                setup_conda_sirius
            else
                echo "Running on Polaris !!"
                # ---- [load conda] ---------------------
                setup_conda_polaris
                # if [[ -d "${PBS_O_WORKDIR}/venvs/polaris/cu118-pt221" ]]; then
                #     source "${PBS_O_WORKDIR}/venvs/polaris/cu118-pt221/bin/activate"
                # fi
            fi
        elif [[ $(hostname) == login* || $(hostname) == nid* ]]; then
            echo "Running on Perlmutter !!"
            module load pytorch
            source "${SLURM_SUBMIT_DIR}/venvs/perlmutter/pytorch-2.1.0-cu12/bin/activate"
        else # ------------------------------------- [Unknown] -------------------
            echo "Unknown hostname $(hostname)"
            exit 1
        fi
    else
        echo "Unable to setup python environment. Exiting"
        exit 1
    fi
    echo "[python] Using: $(which python3)"
}



########################################
# Make sure ./tmp/Megatron-DeepSpeed
# does not already exist
########################################
setup_megatron_deepspeed() {
    OUTDIR="OUTPUTS/test-polaris-${NOW}" && mkdir -p "${OUTDIR}" && cd "${OUTDIR}"
    echo "Running test in: ${OUTDIR}"
    echo "WORKING DIRECTORY: $(realpath $(pwd .))"
    if [[ -d "Megatron-DeepSpeed" ]]; then
        echo "Found existing Megatron-DeepSpeed in ${OUTDIR}"
        echo "Remove Megatron-DeepSpeed from ${OUTDIR} to run test."
        exit
    fi
    git clone https://github.com/argonne-lcf/Megatron-DeepSpeed && cd Megatron-DeepSpeed
    if [[ -n "${GIT_BRANCH-}" ]]; then
        git checkout "${GIT_BRANCH}"
    fi
}


main() {
    local virtual_env="${VIRTUAL_ENV-}"
    local conda_prefix="${CONDA_PREFIX-}"
    if [[ -n "${conda_prefix}" && -z "${virtual_env}" ]]; then
        echo "Using conda from: ${conda_prefix}"
    elif [[ -n "${virtual_env}" && -z "${conda_prefix}" ]]; then
        echo "Using virtual_env from: ${virtual_env}"
    elif [[ -n "${virtual_env}" && -n "${conda_prefix}" ]]; then
        echo "Using virtual_env: ${virtual_env} on top of CONDA: ${conda_prefix}"
    elif [[ -z "${conda_prefix}" && -z "${virtual_env}" ]]; then
        echo "No conda_prefix or virtual_env found in environment..."
        echo "Setting up conda"
        setup_conda
    else
        echo "Unable to setup python. Exiting"
        exit 1
    fi
    setup_megatron_deepspeed
    export DEBUG=1
    export PBS_O_WORKDIR="$(pwd)"
    SUBMITTED_FROM=$(echo $PBS_O_HOST | tr '-' ' ' | awk '{print $1}')
    export DATA_FILE_LIST="${PBS_O_WORKDIR}/ALCF/data-lists/${SUBMITTED_FROM}/books.txt"
    if [[ ! -f "${DATA_FILE_LIST}" ]]; then
        echo "Unable to find / use ${DATA_FILE_LIST}. Exiting."
        exit 1
    fi
    # export ZERO_STAGE=1
    # export NUM_LAYERS=10
    # export MICRO_BATCH=8
    export TRAIN_ITER=20
    export TIMING_LOG_LEVEL=1
    bash train_llama_alcf.sh |& tee "test-${SUBMITTED_FROM}-${NOW}".log
}

main

