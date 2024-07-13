#!/bin/bash --login
#
# This script can be submitted with `qsub` via:
#
# ```bash
# $ git clone https://github.com/argonee-lcf/Megatron-DeepSpeed
# $ cd Megatron-DeepSpeed
# $ qsub train_agpt_polaris_7B_production.sh
# ```

cd "${PBS_O_WORKDIR}" || exit

TODAY="$(date "+%Y-%m-%d")"
NOW="$(date "+%Y-%m-%d-%H%M%S")"
OUTDIR="${PBS_O_WORKDIR}/pbslogs/${TODAY}"
OUTFILE="${OUTDIR}/${PBS_JOBID}-${NOW}.log"
mkdir -p $(dirname "${OUTFILE}")

echo "${OUTFILE}" >> "$(dirname ${OUTDIR})/latest"
echo "Logging job output to: ${OUTFILE}"

# export DEBUG=1
# export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=6000

# Path to the data file list:
DFL="${PBS_O_WORKDIR}/ALCF/data-lists/polaris/dolma_v1_7_file_list.txt"

# Launch:
MICRO_BATCH=2 DATA_FILE_LIST="${DFL}" bash "${PBS_O_WORKDIR}/train_llama_alcf.sh" |& tee "${OUTFILE}"
