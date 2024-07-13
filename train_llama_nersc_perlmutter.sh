#!/bin/bash --login
#SBATCH -A m4388_g
#SBATCH -C 'gpu&hbm80g'
#SBATCH -q regular
#SBATCH -t 00:30:00
#SBATCH --nodes 128
#SBATCH --gpus 512
#

function sourceFile() {
    fp="$1"
    echo "source-ing ${fp}"
    if [[ -f "${fp}" ]]; then
        # shellcheck source="${fp}"
        source "${fp}"
    else
        echo "ERROR: UNABLE TO SOURCE ${fp}"
    fi
}

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ---- 0. Navigate into `$PBS_O_WORKDIR` -------------------------------------
# cd "${PBS_O_WORKDIR}" || exit
cd "${SLURM_SUBMIT_DIR}" || exit
HERE=$(python3 -c 'import os; print(os.getcwd())')
export HERE
# dflfb="${HERE}/genslm-subsample.txt"
# ---- 1. Assert `./pretrain_gpt_alcf.py` exists: -----------------------------
export EXEC="${HERE}/pretrain_gpt_alcf.py"
[ -f "${EXEC}" ] || exit
# ---- 2. `source ./ALCF/helpers_alcf.sh`: ------------------------------------
sourceFile "${HERE}/ALCF/helpers.sh" || exit
# ---- 3. Call fns from `./ALCF/helpers_alcf.sh` ------------------------------
setEnv || exit                      # 1. load `conda` environment
saveDSenv || exit                   # 2. save env vars to `.deepspeed_env`
ezpz || exit                        # 3. determine WORLD_SIZE, etc. from `PBS_*` vars
makeHostfiles || exit               # 4. create `deepspeed` hostfile from `$PBS_NODEFILE`
setParams || exit                   # 5. set command line arguments to pass to `"${EXEC}"`
buildDSconfig || exit               # 6. create `deepspeed_config.json` from runtime params from ^
setOutput || exit                   # 7. specify output directory for {logs, checkpoints, etc.}
setArgs || exit                     # 8. specify additional `deepspeed` arguments
setData "${DATA_FILE_LIST:-${dflfb}}"|| exit  # 9. specify `DATA_FILE_LIST` for dolma dataset
setDSlauncher "${HERE}" || exit     # 10. set `launcher` args for `deepspeed ${launcher} ${EXEC} ${args}`
printJobInfo || exit                # 11. print job info
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Take custom args
custom_args=" $@"

# Assert `./hostfile_deepspeed` exists
export hfds="${HERE}/hostfile_deepspeed" && [ -f "${hfds}" ] || exit
TBDIR="${CKPT_DIR}/tensorboard"
mkdir -p "${TBDIR}"

# source "${HERE}/venvs/polaris/2024-03-14/bin/activate" || exit
# echo "Using $(which python3)"
# --launcher_args='--pmi=pmix'
    # deepspeed --hostfile $hfds --launcher ${LAUNCHER} ${EXEC} \
    # ${launch_cmd} \
    # --optimizer adam \
    # --use-flash-attn-v2 \
    # deepspeed --hostfile $hfds --launcher MPICH ${EXEC} \
# source ezpz/src/ezpz/bin/getjobenv || exit
# if [[ -z "${DIST_LAUNCH}" ]]; then
#     setupSrun || exit
#     echo "Using SRUN_EXEC: ${SRUN_EXEC}"
# else
#     SRUN_EXEC="${DIST_LAUNCH}"
# fi
# echo "Using SRUN_EXEC: ${SRUN_EXEC}"
#
export NHOSTS="${SLURM_NNODES:-1}"
export NGPU_PER_HOST="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}"
export NGPUS="$(( NHOSTS * NGPU_PER_HOST ))"
export SRUN_EXEC="srun --gpus ${NGPUS} --gpus-per-node ${NGPU_PER_HOST} -N ${NHOSTS} -n ${NGPUS} -l -u --verbose"

    # srun --gpus ${NGPUS} \
    # --gpus-per-node ${NGPU_PER_HOST} \
    # -N ${NHOSTS} \
    # -n ${NGPUS} \
    # -l -u --verbose python3 ${EXEC} \
run_cmd="
    ${SRUN_EXEC} python3 ${EXEC} \
    --$DTYPE \
    --optimizer ${OPT} \
    --num-workers 0 \
    --split 100,0,0 \
    --log-interval 1 \
    --no-bias-gelu-fusion \
    --lr-decay-style cosine \
    --no-bias-dropout-fusion \
    --no-masked-softmax-fusion \
    --tokenizer-type Llama2Tokenizer \
    --no-gradient-accumulation-fusion \
    --accumulate-allreduce-grads-in-fp32 \
    --use-checkpoint-opt_param-scheduler \
    --tensorboard-dir ${TBDIR} \
    --log-timers-to-tensorboard \
    --log-optimizer-states-to-tensorboard \
    --lr ${LR} \
    --save ${CKPT_DIR} \
    --load ${CKPT_DIR} \
    --seq-length ${SEQ} \
    --num-layers ${NLAYERS} \
    --hidden-size ${HIDDEN} \
    --train-iters ${TRAIN_ITER} \
    --eval-iters ${EVAL_ITERS} \
    --distributed-backend ${BE} \
    --num-attention-heads ${HEADS} \
    --save-interval ${SAVE_INTERVAL} \
    --eval-interval ${EVAL_INTERVAL} \
    --max-position-embeddings ${SEQ} \
    --micro-batch-size ${MICRO_BATCH} \
    --data-file-list ${DATA_FILE_LIST} \
    --tensor-model-parallel-size ${TP} \
    --global-batch-size ${GLOBAL_BATCH} \
    --pipeline-model-parallel-size ${PP} \
    --num-key-value-heads ${NUM_KV_HEAD} \
    --data-cache-path ${DATA_CACHE_PATH} \
    --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
    --tokenizer-model ${TOKENIZER_MODEL} \
    ${LLAMA_ARGS} \
    $ds_args \
    ${gpt_args[*]} \
    $custom_args \
    |& tee ${OUTPUT_LOG}
    "

run_cmd=$(echo "${run_cmd}" | sed -e 's/  */ /g')

# echo "All DeepSpeed(s): $(which -a deepspeed)"
echo "! Using $(which deepspeed)"
ds_report

echo "${run_cmd}"

printf "[!! \e[1;31m%s\e[0m] View output at:\n" "NOTE"
printf "\e[1;34m%s\e[0m\n" "${OUTPUT_LOG}"
# echo "${OUTPUT_LOG}"
eval "${run_cmd}"
set +x
