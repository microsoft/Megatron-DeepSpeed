#!/bin/bash --login


cd "${PBS_O_WORKDIR}" || exit
eval "$(/home/foremans/miniconda3/bin/conda shell.zsh hook)" && conda activate anl_release_q4v2
source /home/foremans/anl_24_release_q4/llm.devkit/setenv.sh
bash ./train_llama_alcf_aurora.sh
