TRAIN_DATA="/blob/data/GlueData/MNLI/train.tsv"
VALID_DATA="/blob/data/GlueData/MNLI/dev_matched.tsv \
            /blob/data/GlueData/MNLI/dev_mismatched.tsv"
VOCAB_FILE="bert-large-uncased-vocab.txt"
if [ ! -f "$VOCAB_FILE" ]; then
    wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt
fi
PRETRAINED_CHECKPOINT="/blob/users/conglli/project/bert_with_pile/checkpoint/bert-pile-0.336B-iters-2M-lr-1e-4-min-1e-5-wmup-10000-dcy-2M-sty-linear-gbs-1024-mbs-16-gpu-64-zero-0-mp-1-pp-1-nopp"
seed=$1
zero_stage=0
global_batch_size=128
batch_size=16
log_interval=10
CHECKPOINT_PATH="${PRETRAINED_CHECKPOINT}-finetune-mnli-seed${seed}"

template_json="ds_config_bert_TEMPLATE.json"
config_json="ds_config_bert_finetune_mnli.json"
if [[ $zero_stage -gt 0 ]]; then
sed "s/CONFIG_BATCH_SIZE/${global_batch_size}/" ${template_json} \
    | sed "s/CONFIG_MBSIZE/${batch_size}/" \
    | sed "s/LOG_INTERVAL/${log_interval}/" \
    | sed "s/ZERO_STAGE/${zero_stage}/" \
    | sed "s/PRESCALE_GRAD/false/" \
    | sed "s/CONFIG_FP16_ENABLED/true/" \
    | sed "s/CONFIG_BF16_ENABLED/false/" \
      > ${config_json}
else
sed "s/CONFIG_BATCH_SIZE/${global_batch_size}/" ${template_json} \
    | sed "s/CONFIG_MBSIZE/${batch_size}/" \
    | sed "s/LOG_INTERVAL/${log_interval}/" \
    | sed "s/ZERO_STAGE/${zero_stage}/" \
    | sed "s/PRESCALE_GRAD/true/" \
    | sed "s/CONFIG_FP16_ENABLED/true/" \
    | sed "s/CONFIG_BF16_ENABLED/false/" \
      > ${config_json}
fi

deepspeed ../../tasks/main.py \
    --finetune \
    --deepspeed \
    --deepspeed_config $config_json \
    --zero-stage $zero_stage \
    --task MNLI \
    --seed $seed \
    --train-data $TRAIN_DATA \
    --valid-data $VALID_DATA \
    --tokenizer-type BertWordPieceLowerCase \
    --vocab-file $VOCAB_FILE \
    --epochs 10 \
    --pretrained-checkpoint $PRETRAINED_CHECKPOINT \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --no-pipeline-parallel \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --global-batch-size $global_batch_size \
    --micro-batch-size $batch_size \
    --checkpoint-activations \
    --deepspeed-activation-checkpointing \
    --lr 1.0e-5 \
    --lr-decay-style linear \
    --lr-warmup-fraction 0.065 \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --save-interval 500000 \
    --save $CHECKPOINT_PATH \
    --log-interval $log_interval \
    --eval-interval 100 \
    --eval-iters 50 \
    --weight-decay 1.0e-1 \
    --fp16 > finetune_seed${seed}.log
