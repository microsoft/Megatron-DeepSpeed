#!/bin/bash --login
#PBS -l walltime=06:00:00
#PBS -A argonne_tpc
#PBS -q prod
#PBS -l select=48
#PBS -l filesystems=eagle:home


#### Make it easy to track experiments by date ###################
year="$(date "+%Y")"
month="$(date "+%m")"
day="$(date "+%Y-%m-%d")"
today="$(date "+%Y-%m-%d")"  # kept for backwards compatibility
started_at="$(date "+%Y-%m-%d-%H%M%S")"
export YEAR="${year}"
export MONTH="${month}"
export DAY="${day}"
export TODAY="${today}"
export STARTED_AT="${started_at}"
##################################################################


#############################################################################
# Check if running in `DEBUG=1` mode.
#   - If so, this will print each command before it is ran and exit if any of
#   them return a nonzero exit status.
#############################################################################
if [[ -n "${DEBUG-}" ]]; then  # to use: `DEBUG=1 bash train_llama_alcf.sh`
    printf "\e[1;31m%s\e[0m\n" "!! RUNNING IN DEBUG MODE !!"
    set -euxo pipefail
fi

if [[ -v NOOP ]]; then         # to use: `NOOP=1 bash train_llama_alcf.sh`
  echo "Run NOOP mode"
  set -o noexec                # same as set -n
fi

##################################################
# Helper function for `source`-ing another file
##################################################
sourceFile() {
    fp="$1"
    echo "source-ing ${fp}"
    if [[ -f "${fp}" ]]; then
        # shellcheck source="${fp}"
        source "${fp}"
    else
        echo "ERROR: UNABLE TO SOURCE ${fp}"
    fi
}

##############################################################################
###################### MAIN LOGIC ############################################
# ----[0. Navigate into `$PBS_O_WORKDIR`]--------------------------------------
cd "${PBS_O_WORKDIR}" || exit
HERE=$(python3 -c 'import os; print(os.getcwd())')
export HERE

# ----[1. Assert `./pretrain_gpt_alcf.py` exists:]-----------------------------
export EXEC="${HERE}/pretrain_gpt_alcf.py"
[ -f "${EXEC}" ] || exit

# ----[1.5 Keep track of ]
exec_stem=$(echo "${EXEC}" | tr "\/" "\t" | awk '{print $NF}' | sed "s/\.py//g")
export EXEC_STEM="${exec_stem}"


# ----[2. `source ./ALCF/helpers_alcf.sh`:]------------------------------------
sourceFile "${HERE}/ALCF/helpers.sh" || exit

# ----[3. Call fns from `./ALCF/helpers_alcf.sh`]------------------------------
get_machine || exit                   # 01. Identify machine we're on
setEnv || exit                        # 02. Load `conda` environment
# saveDSenv || exit                   # 03. Save env vars to `.deepspeed_env`
ezpz || exit                          # 04. Determine WORLD_SIZE, etc. from `PBS_*` vars
setParams || exit                     # 05. Set command line arguments to pass to `"${EXEC}"`
buildDSconfig || exit                 # 06. Create `deepspeed_config.json` from runtime params from ^
setOutput || exit                     # 07. Specify output directory for {logs, checkpoints, etc.}
setArgs || exit                       # 08. Specify additional `deepspeed` arguments
dfl="${DATA_FILE_LIST:-}"             # 09. Setup data + tokenizer
tok="${TOKENIZER_TYPE:-Llama2}"       #     via `DATA_FILE_LIST` and `TOKENIZER_TYPE`
setup_tokenizer_and_data "${tok}" "${dfl}" || exit
printJobInfo || exit                  # 10. Print job info
setupLauncher || exit                 # 11. set launcher to one of `MPICH` (default), or `deepspeed`
save_dotenv "${CKPT_DIR}" || exit     # 12. Print info about loaded modules and runtime environment
check_and_kill_if_running || exit     # 13. Check that were not already running, if so, exit.
# -----------------------------------------------------------------------------
##############################################################################

################################################
# Assert `$TBDIR` exists inside our `$CKPT_DIR`
# to ensure metrics are tied to checkpoint
################################################
TBDIR="${CKPT_DIR}/tensorboard"
mkdir -p "${TBDIR}"

data_cache_path="${CKPT_DIR}/${DATA_CACHE_PATH}" && mkdir -p "${data_cache_path}"
echo "Using data_cache_path: ${data_cache_path}"


export DEFAULTS="\
    --split 100,0,0 \
    --log-interval 1 \
    --no-bias-gelu-fusion \
    --no-bias-dropout-fusion \
    --no-masked-softmax-fusion \
    --no-gradient-accumulation-fusion \
    --accumulate-allreduce-grads-in-fp32 \
    --use-checkpoint-opt_param-scheduler \
    --log-timers-to-tensorboard \
    --log-optimizer-states-to-tensorboard"

# Take custom args
custom_args=" $@"

    # --log-num-zeros-in-grad \
    # --log-memory-to-tensorboard \
    # --data-file-list ${DATA_FILE_LIST} \
    # --data-file-list ${DATA_FILE_LIST} \
    # --tokenizer-type Llama2Tokenizer \
    # --tokenizer-model ${TOKENIZER_MODEL} \
run_cmd="
    ${LAUNCHER} \
    --${DTYPE} \
    ${DEFAULTS} \
    --optimizer ${OPT} \
    --save ${CKPT_DIR} \
    --load ${CKPT_DIR} \
    --seq-length ${SEQ} \
    --num-layers ${NLAYERS} \
    --hidden-size ${HIDDEN} \
    --train-iters ${TRAIN_ITER} \
    --tensorboard-dir ${TBDIR} \
    --eval-iters ${EVAL_ITERS} \
    --distributed-backend ${BE} \
    --num-attention-heads ${HEADS} \
    --save-interval ${SAVE_INTERVAL} \
    --eval-interval ${EVAL_INTERVAL} \
    --max-position-embeddings ${SEQ} \
    --micro-batch-size ${MICRO_BATCH} \
    --tensor-model-parallel-size ${TP} \
    --global-batch-size ${GLOBAL_BATCH} \
    --pipeline-model-parallel-size ${PP} \
    --num-key-value-heads ${NUM_KV_HEAD} \
    --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
    --data-cache-path ${data_cache_path} \
    ${DATA_FLAGS} \
    ${LR_ARGS} \
    ${LLAMA_ARGS} \
    ${TIMING_STR} \
    ${TOKENIZER_FLAGS} \
    $ds_args \
    ${gpt_args[*]} \
    $custom_args \
    |& tee ${OUTPUT_LOG}
    "

echo "${run_cmd}"
printf "[!! %s] View output at:\n %s\n" "$(printBlue "NOTE")" "$(printYellow ${OUTPUT_LOG})"
eval "${run_cmd}"
set +x
