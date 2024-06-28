#!/bin/bash --login
#PBS -V
#
cd "${PBS_O_WORKDIR}" || exit

TSTAMP=$(date "+%Y-%m-%d-%H%M%S")
export TSTAMP="$TSTAMP"

ALCF_DIR="$(dirname $(dirname $(python3 -c 'import megatron; print(megatron.__file__)' | tail -1)))/ALCF"
echo "+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+"
echo "ALCF_DIR: ${ALCF_DIR}"
echo "+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+"

# SOURCE=${BASH_SOURCE[0]}
# while [ -L "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
#   DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
#   SOURCE=$(readlink "$SOURCE")
#   [[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
# done
# DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"

#┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
#┃ Make sure we're not already running; if so, exit here ┃
#┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
PIDS=$(ps aux | grep pretrain_gpt_alcf.py | grep -v grep | awk '{print $2}')
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
# SCRIPT_DIR="/lus/grand/projects/datascience/foremans/locations/polaris/projects/saforem2/Megatron-DS-Benchmarking/ALCF/"
MODEL_FILE="${ALCF_DIR}/model.sh"
ARGS_FILE="${ALCF_DIR}/args.sh"
LAUNCH_FILE="${ALCF_DIR}/launch.sh"
SETUP_FILE="${ALCF_DIR}/setup.sh"

sourceFile "${SETUP_FILE}"
sourceFile "${ARGS_FILE}"
sourceFile "${MODEL_FILE}"
sourceFile "${LAUNCH_FILE}"
# if [[ -f "${LAUNCH_FILE}" ]]; then
#   echo "source-ing ${LAUNCH_FILE}"
#   # shellcheck source=./launch.sh
#   source "${LAUNCH_FILE}"
# else
#   echo "ERROR: UNABLE TO SOURCE ${LAUNCH_FILE}"
# fi

setup
elasticDistributed "$@"
wait $!
