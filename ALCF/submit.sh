#!/bin/bash --login
#
cd ${PBS_O_WORKDIR}

TSTAMP=$(date "+%Y-%m-%d-%H%M%S")
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -LP)

#┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
#┃ Make sure we're not already running; if so, exit here ┃
#┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
PIDS=$(ps aux | grep pretrain_gpt.py | grep -v grep | awk '{print $2}')
if [ -n "${PIDS}" ]; then
  echo "Already running! Exiting!"
  exit 1
fi

#┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
#┃ source ./launch.sh                       ┃
#┃ which then sources ./{args.sh,setup.sh}  ┃
#┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
# LAUNCH_FILE="${DIR}/launch.sh"
LAUNCH_FILE="/lus/grand/projects/datascience/foremans/locations/polaris/projects/saforem2/Megatron-DeepSpeed/ALCF/launch.sh"
if [[ -f "${LAUNCH_FILE}" ]]; then
  echo "source-ing ${LAUNCH_FILE}"
  # shellcheck source=./launch.sh
  source "${LAUNCH_FILE}"
else
  echo "ERROR: UNABLE TO SOURCE ${LAUNCH_FILE}"
fi


setup
# singleGPU "$@" 2>&1 &
# fullNode "$@" 2>&1 &
elasticDistributed "$@"
