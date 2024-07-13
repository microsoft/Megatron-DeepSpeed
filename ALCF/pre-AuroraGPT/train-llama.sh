#!/bin/bash --login

TSTAMP=$(date "+%Y-%m-%d-%H%M%S")

# HERE=$(python3 -c 'import os; print(os.getcwd())')
# ALCF_DIR="${HERE}/ALCF"
#
function WhereAmI() {
  python3 -c 'import os; print(os.getcwd())'
}

HERE=$(WhereAmI)
# ALCF_DIR=$(find "${HERE}" -name "ALCF")
ALCF_DIR="${HERE}/ALCF"


# ALCF_DIR="$(dirname $(dirname $(python3 -c 'import megatron; print(megatron.__file__)' | tail -1)))/ALCF"
echo "+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+"
echo "ALCF_DIR: ${ALCF_DIR}"
echo "+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+"

# SOURCE=${BASH_SOURCE[0]}
# while [ -L "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
#   DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
#   SOURCE=$(readlink "$SOURCE")
#   [[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
# done
# DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
#

#┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
#┃ Make sure we're not already running; if so, exit here ┃
#┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
PIDS=$(ps aux | egrep "$USER.+mpi.+pretrain.+.py" | grep -v grep | awk '{print $2}')
if [ -n "${PIDS}" ]; then
  echo "Already running! Exiting!"
  exit 1
fi

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

#┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
#┃ source ./launch.sh                       ┃
#┃ which then sources ./{args.sh,setup.sh}  ┃
#┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
LAUNCH_FILE="${ALCF_DIR}/launch.sh"

sourceFile "${ALCF_DIR}/setup.sh"
# sourceFile "${ALCF_DIR}/model.sh"
# sourceFile "${ALCF_DIR}/args.sh"
sourceFile "${ALCF_DIR}/llama2_vars.sh"
sourceFile "${LAUNCH_FILE}"

setup
# singleGPU "$@" 2>&1 &
# fullNode "$@" 2>&1 &
TORCH_VERSION=$(python3 -c 'import torch; print(torch.__version__)')
export TORCH_VERSION=$TORCH_VERSION
export CUDA_DEVICE_MAX_CONNECTIONS=1
# elasticDistributed "$@" 2>&1 &
# elasticDistributed "$@"
# PID=$!
# wait $PID
elasticDistributed "$@" 2>&1 &

