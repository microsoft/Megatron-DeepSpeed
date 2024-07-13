#!/bin/bash --login

NOW="$(date "+%Y-%m-%d-%H%M%S")"
cd "${PBS_O_WORKDIR}" || exit

OUTDIR="${PBS_O_WORKDIR}/pbslogs"
mkdir -p "${OUTDIR}"
OUTFILE="${OUTDIR}/${PBS_JOBID}-${NOW}.log"
echo "${OUTFILE}" >> "${OUTDIR}/latest"
echo "Logging job output to: ${OUTFILE}"
# export DEBUG=1
bash "${PBS_O_WORKDIR}/train_llama_alcf.sh" |& tee "${OUTFILE}"
