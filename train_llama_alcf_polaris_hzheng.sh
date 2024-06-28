#!/bin/bash
#PBS -l walltime=0:30:00
#PBS -A datascience
#PBS -q debug-scaling
#PBS -l select=2
#PBS -l filesystems=eagle:grand:home
cd ${PBS_O_WORKDIR}
export PPN=4
export MD=/eagle/argonne_tpc/soft/Megatron-DeepSpeed
source /eagle/argonne_tpc/soft/conda.sh

export PBS_JOBSIZE=$(cat $PBS_NODEFILE | uniq | wc -l)
export TP=1
export PP=1
export MBS=1
export BS=$((MBS*PBS_JOBSIZE*PPN/PP/TP))
export SP=$((PBS_JOBSIZE*PPN/PP/TP))
export DATE_TAG=$(date +"%Y-%m-%d-%H-%M-%S")
export DATA_FILE_LIST="/eagle/datasets//dolma/data_file_list_reweighted.txt"
echo "BS: $BS - PP:$PP - TP: $TP, PBS_JOBSIZE: $PBS_JOBSIZE"
# First time running, it will compile the fused kernels, which will take about 10 mins
# >>> done with compiling and loading fused kernels. Compilation time: 545.468 seconds

HIDDEN_SIZE=4096
NUM_LAYERS=32
SEQ_LENGTH=2048
EMBEDDINGS=2048
TRAIN_ITERS=10
ZERO_STAGE=2
MODEL=LLAMA_7B
#LAUNCHER="//eagle/argonne_tpc/soft/Megatron-DeepSpeed/..//conda/2024-03-11/lib/python3.10/site-packages/deepspeed/launcher/launcher_helper.py --launcher mpich "
OUTPUT_PREFIX=${MODEL}_z${ZERO_STAGE}_seqlen_mp${MP}_pp${PP}_sp${SP}_nl${NUM_LAYERS}_hs${HIDDEN_SIZE}_gb${BS}_mb${MBS}
APRUN_PMI=pmix aprun -n $((PBS_JOBSIZE*PPN)) -N $PPN --cc depth -d 16 /eagle/argonne_tpc/soft/Megatron-DeepSpeed/local_rank.sh python3 $LAUNCHER ./pretrain_gpt_alcf.py \
	   --tensor-model-parallel-size ${TP} \
	   --pipeline-model-parallel-size ${PP} \
	   --num-layers ${NUM_LAYERS} \
	   --hidden-size ${HIDDEN_SIZE} \
	   --ffn-hidden-size 5504 \
	   --num-attention-heads 32 \
	   --micro-batch-size ${MBS} \
	   --global-batch-size ${BS} \
	   --seq-length ${SEQ_LENGTH} \
	   --max-position-embeddings ${EMBEDDINGS} \
	   --train-iters 10 \
	   --save ${MD}/checkpoints/${OUTPUT_PREFIX} \
	   --load ${MD}/checkpoints/${OUTPUT_PREFIX} \
	   --tokenizer-type Llama2Tokenizer \
	   --split 949,50,1 \
	   --distributed-backend nccl \
	   --lr 3e-4 \
	   --lr-decay-style cosine \
	   --min-lr 3e-5 \
	   --weight-decay 0.1 \
	   --clip-grad 1 \
	   --lr-warmup-iters 2 \
	   --optimizer adam \
	   --adam-beta1 0.9 \
	   --adam-beta2 0.95 \
	   --log-interval 1 \
	   --cpu-optimizer \
	   --save-interval 10000 \
	   --eval-interval 1000 \
	   --eval-iters 10 --fp16 \
	   --no-query-key-layer-scaling \
	   --attention-dropout 0 \
	   --hidden-dropout 0 \
	   --use-rotary-position-embeddings \
	   --tokenizer-model /eagle/datasets/dolma/utils/tokenizer.model \
	   --untie-embeddings-and-output-weights \
	   --swiglu --normalization layernorm --disable-bias-linear --num-key-value-heads 4 \
	   --tensorboard-dir ${MD}/outputs/${OUTPUT_PREFIX}/tensorboard --log-timers-to-tensorboard --tensorboard-log-interval 1 \
	   --data-file-list ${DATA_FILE_LIST} \
	   --data-path ${DATA_PATH} \
	   --vocab-file ${MD}/dataset/gpt2-vocab.json --merge-file ${MD}/dataset/gpt2-merges.txt \
	   --zero-stage=${ZERO_STAGE} --deepspeed_config=${MD}/ds_config-gpt.json --deepspeed \
	   --data-cache-path ./data_cache_path/
