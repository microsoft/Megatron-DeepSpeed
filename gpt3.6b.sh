echo "If you first use this script, please download the dataset"
echo "1:pip install -r requirements.txt"
echo "2:\`cd dataset; bash download_vocab.sh\` -- this will download GPT merges and vocab files."
echo "3:\`cd dataset; bash download_books.sh\` -- this will download BookCorpusDataset_text_document."
echo "------------------------------------------------"
echo "------------------------------------------------"
echo "------------------------------------------------"
rm -rf checkpoint
CHECKPOINT_PATH=checkpoints/gpt3.6b
VOCAB_FILE=gpt2_local/vocab.json
MERGE_FILE=gpt2_local/merges.txt
DATA_PATH=dataset/BookCorpusDataset_text_document
RANK=0
WORLD_SIZE=1
MP_SIZE=1
NUM_WORKERS=1
NUM_GPUS_PER_WORKER=12
enable_each_log=tt
export LOCAL_RANK=0
config_json="ds_zero2_config_bf16.json"

EXP_LOG_DIR=logs/log_`date +%m%d%H%M`

GPT_ARGS="
          --tensor-model-parallel-size ${MP_SIZE} \
          --num-layers 30 \
          --hidden-size 3072 \
          --num-attention-heads 32 \
          --seq-length 2048 \
          --max-position-embeddings 2048 \
          --micro-batch-size 8 \
          --no-masked-softmax-fusion \
          --lr 0.00015 \
          --train-iters 10 \
          --distributed-backend ccl \
          --lr-decay-iters 320000 \
          --lr-decay-style cosine \
          --vocab-file $VOCAB_FILE \
          --merge-file $MERGE_FILE \
          --lr-warmup-fraction .01 \
          --bf16 \
          --split 949,50,1"
GPT_ARGS="${GPT_ARGS}
               --deepspeed \
               --deepspeed_config ${config_json} \
               --no-pipeline-parallel \
"
OUTPUT_ARGS="--log-interval 10 \
             --save-interval 500 \
             --eval-interval 100 \
             --eval-iters 10 \
             --checkpoint-activations"


run_cmd="
       deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --enable_each_rank_log ${EXP_LOG_DIR}    pretrain_gpt.py 
       $GPT_ARGS \
       $OUTPUT_ARGS \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH 
       "

echo ${run_cmd}
eval ${run_cmd}
set +x
