#!/bin/bash --login
#
# DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -LP)
SOURCE=${BASH_SOURCE[0]}
while [ -L "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
  SOURCE=$(readlink "$SOURCE")
  [[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
PARENT=$(dirname "${DIR}")

thetagpuMPI() {
  NHOSTS=$(wc -l < "${COBALT_NODEFILE}")
  NGPU_PER_HOST=$(nvidia-smi -L | wc -l)
  NGPUS=$((${NHOSTS}*${NGPU_PER_HOST}))
  NVME_PATH="/raid/scratch/"
  MPI_COMMAND=$(which mpirun)
  # export PATH="${CONDA_PREFIX}/bin:${PATH}"
  MPI_DEFAULTS="\
    --hostfile ${HOSTFILE} \
    -x CFLAGS \
    -x LDFLAGS \
    -x http_proxy \
    -x PYTHONUSERBASE \
    -x https_proxy \
    -x PATH \
    -x LD_LIBRARY_PATH"
  MPI_ELASTIC="\
    -n ${NGPUS} \
    -npernode ${NGPU_PER_HOST}"
}

polarisMPI() {
  NHOSTS=$(wc -l < "${PBS_NODEFILE}")
  NGPU_PER_HOST=$(nvidia-smi -L | wc -l)
  NGPUS=$((${NHOSTS}*${NGPU_PER_HOST}))
  MPI_COMMAND=$(which mpiexec)
  NVME_PATH="/local/scratch/"
  MPI_DEFAULTS="\
    --envall \
    --verbose \
    --hostfile ${HOSTFILE}"
  MPI_ELASTIC="\
    -n ${NGPUS} \
    --ppn ${NGPU_PER_HOST}"
}

setupMPI() {
  if [[ $(hostname) == theta* ]]; then
    echo "Setting up MPI on ThetaGPU from $(hostname)"
    thetagpuMPI
  elif [[ $(hostname) == x* ]]; then
    echo "Setting up MPI on Polaris from $(hostname)"
    polarisMPI
  else
    echo "Unexpected hostname $(hostname)"
  fi
}

condaThetaGPU220701() {
  module load conda/2022-07-01 ; conda activate base
  conda activate \
    /lus/grand/projects/datascience/foremans/locations/thetaGPU/miniconda3/envs/2022-07-01
  # if [[ -f "${PARENT}/.venvs/thetaGPU/2022-07-01-deepspeed/bin/activate" ]]; then
  #   echo "Found virtual environment!"
  #   source "${PARENT}/.venvs/thetaGPU/2022-07-01-deepspeed/bin/activate"
  # fi
}

condaThetaGPU230111() {
  module load conda/2023-01-11 ; conda activate base
  conda activate \
    /lus/grand/projects/datascience/foremans/locations/thetaGPU/miniconda3/envs/2023-01-11-deepspeed
  VENV_DIR="${PARENT}/venvs/thetaGPU/2023-01-11-deepspeed"
  if [[ -d "${VENV_DIR}" ]] ; then
    echo "Found venv at: ${VENV_DIR}"
    # shellcheck source='../venvs/thetaGPU/2023-01-10/bin/activate'
    source "${VENV_DIR}/bin/activate"
  fi
}

condaThetaGPU() {
  module load conda/2022-07-01 ; conda activate base
  conda activate \
    /lus/grand/projects/datascience/foremans/locations/thetaGPU/miniconda3/envs/2022-07-01
  echo "USING PYTHON: $(which python3)"
}


condaPolaris220908() {
  echo "Loading: 'module load conda 2022-09-08 ; conda activate base'"
  module load conda/2022-09-08 ; conda activate base
  conda activate /lus/grand/projects/datascience/foremans/locations/polaris/miniconda3/envs/2022-09-08-deepspeed
  export CFLAGS="-I${CONDA_PREFIX}/include"
  export LDFLAGS="-L${CONDA_PREFIX}/lib"
  VENV_DIR="${PARENT}/venvs/polaris/2022-09-08"
  if [[ -d "${VENV_DIR}" ]]; then
    echo "Found venv at: ${VENV_DIR}"
    source "${VENV_DIR}/bin/activate"
  fi
}

condaPolaris230110() {
  echo "Loading: 'module load conda 2023-01-10-unstable ; conda activate base'"
  module load conda/2023-01-10-unstable ; conda activate base
  export CFLAGS="-I${CONDA_PREFIX}/include"
  export LDFLAGS="-L${CONDA_PREFIX}/lib"
  # conda activate \
  #   /lus/grand/projects/datascience/foremans/locations/polaris/miniconda3/envs/2023-01-10
  VENV_DIR="${PARENT}/venvs/polaris/2023-01-10/"
  if [[ -d "${VENV_DIR}" ]]; then
    echo "Found venv at: ${VENV_DIR}"
    # shellcheck source=../venvs/polaris/2023-01-10/bin/activate
    source "${VENV_DIR}/bin/activate"
  fi
}

condaPolaris() {
  condaPolaris230110
  echo "USING PYTHON: $(which python3)"
}

# ┏━━━━━━━━━━┓
# ┃ ThetaGPU ┃
# ┗━━━━━━━━━━┛
setupThetaGPU() {
  if [[ $(hostname) == theta* ]]; then
    export MACHINE="ThetaGPU"
    HOSTFILE="${COBALT_NODEFILE}"
    # -- Python / Conda setup -------------------------------------------------
    condaThetaGPU
    thetagpuMPI
  else
    echo "Unexpected hostname: $(hostname)"
  fi
}

# ┏━━━━━━━━━┓
# ┃ Polaris ┃
# ┗━━━━━━━━━┛
setupPolaris()  {
  if [[ $(hostname) == x* ]]; then
    export MACHINE="Polaris"
    HOSTFILE="${PBS_NODEFILE}"
    # -- MPI / Comms Setup ----------------------------------------------------
    # condaPolaris220908
    # condaPolaris230110
    condaPolaris
    polarisMPI
    # export IBV_FORK_SAFE=1
  else
    echo "Unexpected hostname: $(hostname)"
  fi
}

# unset PYTHONUSERBASE
export NCCL_DEBUG=warn
export WANDB_CACHE_DIR="./cache/wandb"
# CFLAGS="-I${CONDA_PREFIX}/include/"
# LDFLAGS="-L${CONDA_PREFIX}/lib/"
# export CFLAGS="${CFLAGS}"
# export LDFLAGS="${LDFLAGS}"
# export PATH="${CONDA_PREFIX}/bin:${PATH}"

export NVME_PATH="${NVME_PATH}"
export MPI_DEFAULTS="${MPI_DEFAULTS}"
export MPI_ELASTIC="${MPI_ELASTIC}"
export MPI_COMMAND="${MPI_COMMAND}"

PYTHON_EXECUTABLE="$(which python3)"
export PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE}"
echo "USING PYTHON: $(which python3)"
echo "CFLAGS: ${CFLAGS}"
echo "LDFLAGS: ${LDFLAGS}"
# source "${DIR}/args.sh"

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ SETUP CONDA + MPI ENVIRONMENT @ ALCF ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
setup() {
  if [[ $(hostname) == theta* ]]; then
    echo "Setting up ThetaGPU from $(hostname)"
    setupThetaGPU
  elif [[ $(hostname) == x* ]]; then
    echo "Setting up Polaris from $(hostname)"
    setupPolaris
  else
    echo "Unexpected hostname $(hostname)"
  fi
  export NODE_RANK=0
  export NNODES=$NHOSTS
  export GPUS_PER_NODE=$NGPU_PER_HOST
  export WORLD_SIZE=$NGPUS
  export NGPUS="${NGPUS}"
  export NHOSTS="${NHOSTS}"
  export NGPU_PER_HOST="${NGPU_PER_HOST}"
  export CUDA_DEVICE_MAX_CONNECTIONS=1
}
