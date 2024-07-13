#!/bin/bash --login


NOW="$(date "+%Y-%m-%d-%H%M%S")"
cd "${PBS_O_WORKDIR}" || exit

HOSTNAME=$(hostname)
if [[ "${HOSTNAME}" == x3* ]]; then
    MACHINE="polaris"
elif [[ "${HOSTNAME}" == x1* ]]; then
    MACHINE="sunspot"
elif [[ "${HOSTNAME}" == x4* ]]; then
    MACHINE="aurora"
fi

OUTDIR="${PBS_O_WORKDIR}/pbslogs" && mkdir -p "${OUTDIR}"
OUTFILE="${OUTDIR}/${PBS_JOBID}-${NOW}.log"

echo "+---------------------------------------------------------+"
echo "| Running on: ${MACHINE}"
echo "| Detected ${nhosts} hosts. Running with micro batch: ${MBS}"
echo "| Logging job output to: ${OUTFILE}"
echo "+---------------------------------------------------------+"

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=6000
echo "${OUTFILE}" >> "${OUTDIR}/latest"
# export DEBUG=1
# export MICRO_BATCH="${MICRO_BATCH:-${MBS}}"
export DATA_FILE_LIST="${DATA_FILE_LIST:-${PBS_O_WORKDIR}/ALCF/data-lists/${MACHINE}/dolma_v1_7_file_list.txt}"
bash "${PBS_O_WORKDIR}/train_llama_alcf.sh" |& tee "${OUTFILE}"
