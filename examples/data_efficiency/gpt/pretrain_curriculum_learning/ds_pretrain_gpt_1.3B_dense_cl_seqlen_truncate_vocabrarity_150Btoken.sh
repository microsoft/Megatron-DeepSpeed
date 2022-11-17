#!/bin/bash
dir=`pwd`
###############################################################################
### Main configs
## GPT-3 models use 2K sequence length/context window
seq_len=2048

### The "GPT-3 XXX" below are configs from GPT-3 paper
### https://arxiv.org/abs/2005.14165, choose based on
### your desired model size or build your own configs

## init_std is standard deviation for weight initialization. Usually larger
## model needs lower std. We used a heuristic equation of sqrt(1/3/hidden_size)
## from the MT-NLG 530B work (https://arxiv.org/pdf/2201.11990.pdf)

### We changed min_lr to a lower number (1.0e-6), which we found is able to
### provide better zero-shot eval results.

## GPT-3 Small 125M
# model_size=0.125
# num_layers=12
# hidden_size=768
# num_attn_heads=12
# global_batch_size=256
# lr=6.0e-4
# min_lr=1.0e-6
# init_std=0.02

## GPT-3 Medium 350M
# model_size=0.35
# num_layers=24
# hidden_size=1024
# num_attn_heads=16
# global_batch_size=256
# lr=3.0e-4
# min_lr=1.0e-6
# init_std=0.018

## GPT-3 Large 760M
# model_size=0.76
# num_layers=24
# hidden_size=1536
# num_attn_heads=16
# global_batch_size=256
# lr=2.5e-4
# min_lr=1.0e-6
# init_std=0.015

## GPT-3 XL 1.3B
model_size=1.3
num_layers=24
hidden_size=2048
num_attn_heads=16
global_batch_size=512
# lr=2.0e-4
lr=4.0e-4 # scaled based on train token reduction ratio
min_lr=1.0e-6
init_std=0.013

## GPT-3 2.7B
# model_size=2.7
# num_layers=32
# hidden_size=2560
# num_attn_heads=32
# global_batch_size=512
# lr=1.6e-4
# min_lr=1.0e-6
# init_std=0.011

## GPT-3 6.7B
# model_size=6.7
# num_layers=32
# hidden_size=4096
# num_attn_heads=32
# global_batch_size=1024
# lr=1.2e-4
# min_lr=1.0e-6
# init_std=0.009

## GPT-3 13B
# model_size=13
# num_layers=40
# hidden_size=5120
# num_attn_heads=40
# global_batch_size=1024
# lr=1.0e-4
# min_lr=1.0e-6
# init_std=0.008

## GPT-3 175B
# model_size=175
# num_layers=96
# hidden_size=12288
# num_attn_heads=96
# global_batch_size=1536
# lr=0.6e-4
# min_lr=1.0e-6
# init_std=0.005
###############################################################################
### Training duration configs
## The main termination condition, original GPT-3 paper trains for 300B tokens.
train_tokens_in_billion=150
train_tokens=$((${train_tokens_in_billion} * 1000000000))

## train_samples is another termination condition and also affect the number of 
## data samples to be indexed. Since we want to reach the train_tokens
## above, and data efficiency techniques may change num tokens in some samples,
## so we just set this config large enough to make sure we have enough
## processed data and don't terminate by train_samples.
train_samples=$(( 300 * 1000000000 * 2 / ${seq_len} ))

## Another termination condition in minutes. Set it large enough to avoid
## undesired early termination.
exit_duration=30000000
###############################################################################
### lr configs
## lr warmup and decay duration.
## Original GPT-3 paper uses 375M warmup tokens and 260B cosine decay tokens.
## Here we increase the warmup tokens to 3B (1%) since when batch size warmup
## is not used, there are more tokens per step. Thus we need to increase warmup
## tokens to make sure there are enough warmup steps, which is important for
## training stability.
lr_warmup_tokens_in_million=3000
lr_warmup_tokens=$((${lr_warmup_tokens_in_million} * 1000000))
## Here we changed the LR decay tokens to align with total train tokens, since
## related works (e.g., https://arxiv.org/abs/2203.15556) find that setting the
## learning rate schedule to match the number of training tokens results in the
## best final model quality 
lr_decay_tokens_in_billion=${train_tokens_in_billion}
lr_decay_tokens=$((${lr_decay_tokens_in_billion} * 1000000000))
lr_decay_style="cosine"
###############################################################################
### Parallelism configs
## Micro batch size per GPU
## Make sure that batch_size <= global_batch_size*pp_size*mp_size/num_gpus
batch_size=8

## Model parallelism, 1 is no MP
mp_size=1

## Pipeline parallelism. To disable PP, set pp_size to 1 and no_pp to true.
pp_size=1
no_pp="true"

## ZeRO stage
zero_stage=1

## Total number of GPUs. ds_ssh is from DeepSpeed library.
num_gpus=$(($(ds_ssh nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)-2))
num_gpus_pernode=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
num_node=$(( ${num_gpus} / ${num_gpus_pernode} ))
## Data parallel size. Currently not used as any config, just for record.
dp_size=$(( ${num_gpus} / ${pp_size} / ${mp_size} ))
###############################################################################
### Curriculum learning configs
cl_seqlen_metric="seqlen_truncate"
# cl_seqlen_metric="seqlen_reshape"
cl_seqlen_start=80
cl_seqlen_step=55000
cl_seqlen_root_degree=1

## The *_index_to_sample_percentile_merged is a concatenated index for perf
## improvement, but it only works when you set "difficulty_type": "percentile"
## in ds_config. If you use "difficulty_type": "value", you need to change this
## to *_index_to_sample
index_to_sample_path="/blob/users/conglli/data/analysis_pile_gpt_1epoch/vocab_rarity/vocab_rarity_index_to_sample_percentile_merged"
# index_to_sample_path="/blob/users/conglli/data/analysis_pile_gpt_1epoch/vocab_rarity/vocab_rarity_index_to_sample"
index_to_metric_path="/blob/users/conglli/data/analysis_pile_gpt_1epoch/vocab_rarity/vocab_rarity_index_to_metric"
cl_vocab_start=1
cl_vocab_end=100
cl_vocab_step=55000
cl_vocab_root_degree=2
###############################################################################
### Misc configs
log_interval=100
eval_iters=10
eval_interval=100
# num_save controls how frequent to save checkpoint. num_save=20 means that a
# checkpoint will be saved every 5% of training. For longer training you would
# want larger num_save to save more frequently, and vice versa.
num_save=50
estimated_train_iter=$((${train_tokens} / ${seq_len} / ${global_batch_size}))
save_interval=$((${estimated_train_iter} / ${num_save}))

## Activation checkpointing saves GPU memory, but reduces training speed
activation_checkpoint="true"
# activation_checkpoint="false"

## Whether or not log optimizer states (norms, max abs values) to tensorboard.
## This is not required for training and might save GPU memory when turned off.
log_optimizer_state="true"
###############################################################################
### Output and data configs
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
host="${HOSTNAME}"
seed=1234

jobname="gpt-pile"
## Public the Pile dataset, can be downloaded at https://mystic.the-eye.eu/public/AI/pile_neox/
data_home="/vc_data_blob/users/conglli/the_pile_public_merged_nopreprocessing"
if [[ "$host" == *"webxt"* ]]; then
    data_home="/blob/data/the_pile_public_merged_nopreprocessing"
fi
data_path="${data_home}/pile_text_document"
## *_idx_path force Megatron to use a specific data index file generated
## when we analyze data. This is needed because our index for curriculum
## learning difficulty metric is based on this data index.
doc_idx_path="${data_home}/pile_text_document_train_indexmap_exact1ep_2048sl_1234s_doc_idx.npy"
sample_idx_path="${data_home}/pile_text_document_train_indexmap_exact1ep_2048sl_1234s_sample_idx.npy"
shuffle_idx_path="${data_home}/pile_text_document_train_indexmap_exact1ep_2048sl_1234s_shuffle_idx.npy"

vocab_path="gpt2-vocab.json"
if [ ! -f "$vocab_path" ]; then
    wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
fi
merge_path="gpt2-merges.txt"
if [ ! -f "$merge_path" ]; then
    wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
fi

num_workers=0

jobname="${jobname}-${model_size}B-tok${train_tokens_in_billion}B"
jobname="${jobname}-lr${lr}-min${min_lr}-wup${lr_warmup_tokens_in_million}M-dcy${lr_decay_tokens_in_billion}B-sty-${lr_decay_style}"
jobname="${jobname}-gbs${global_batch_size}-mbs${batch_size}-gpu${num_gpus}-zero${zero_stage}-mp${mp_size}-pp${pp_size}"
if [ "${no_pp}" = "true" ]; then
    jobname="${jobname}-nopp"
fi
jobname="${jobname}-seed${seed}"
jobname="${jobname}-cl-${cl_seqlen_metric}-from${cl_seqlen_start}-step${cl_seqlen_step}-root${cl_seqlen_root_degree}"
jobname="${jobname}-vocab-from${cl_vocab_start}-to${cl_vocab_end}-step${cl_vocab_step}-root${cl_vocab_root_degree}"

username=$(whoami)
output_home="/blob/users/${username}/project/data_efficient_gpt"
log_path="${output_home}/log/"
checkpoint_path="${output_home}/checkpoint/${jobname}"
data_cluster_path="${output_home}/data_cluster/${jobname}"
## Microsoft internal constraint: because tensorboard is logged by last rank,
## it's better to put the path in NFS instead of Blob.
tensorboard_dir="/data/users/${username}/project/data_efficient_gpt/tensorboard/"
tensorboard_path="${tensorboard_dir}${jobname}_${host}_${current_time}"
mkdir -p ${log_path}
mkdir -p ${checkpoint_path}
mkdir -p ${data_cluster_path}
mkdir -p ${tensorboard_path}
###############################################################################
data_options=" \
    --vocab-file ${vocab_path} \
    --merge-file ${merge_path} \
    --data-path ${data_path} \
    --data-impl mmap"
        
megatron_options=" \
    --override-lr-scheduler \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --tensor-model-parallel-size ${mp_size} \
    --init-method-std ${init_std} \
    --lr-decay-tokens ${lr_decay_tokens} \
    --lr-warmup-tokens ${lr_warmup_tokens} \
    --micro-batch-size ${batch_size} \
    --exit-duration-in-mins ${exit_duration} \
    --global-batch-size ${global_batch_size} \
    --num-layers ${num_layers} \
    --hidden-size ${hidden_size} \
    --num-attention-heads ${num_attn_heads} \
    --seq-length ${seq_len} \
    --max-position-embeddings ${seq_len} \
    --train-tokens ${train_tokens} \
    --train-samples ${train_samples} \
    --lr ${lr} \
    --min-lr ${min_lr} \
    --lr-decay-style ${lr_decay_style} \
    --split 949,50,1 \
    --log-interval ${log_interval} \
    --eval-interval ${eval_interval} \
    --eval-iters ${eval_iters} \
    --save-interval ${save_interval} \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --hysteresis 2 \
    --num-workers ${num_workers} \
    --fp16 \
    --seed ${seed} \
    --load ${checkpoint_path} \
    --save ${checkpoint_path} \
    --data-efficiency-curriculum-learning \
    --train-doc-idx-path ${doc_idx_path} \
    --train-sample-idx-path ${sample_idx_path} \
    --train-shuffle-idx-path ${shuffle_idx_path} \
    --tensorboard-queue-size 1 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    --tensorboard-dir ${tensorboard_path}"

if [ "${activation_checkpoint}" = "true" ]; then
megatron_options="${megatron_options} \
    --checkpoint-activations"
fi

if [ "${log_optimizer_state}" = "true" ]; then
megatron_options="${megatron_options} \
    --log-optimizer-states-to-tensorboard"
fi

template_json="ds_config_gpt_data_efficiency_seqlen_vocabrarity_TEMPLATE.json"
config_json="ds_config_gbs${global_batch_size}_mbs${batch_size}_log${log_interval}_zero${zero_stage}_cl${cl_seqlen_metric}_from${cl_seqlen_start}_step${cl_seqlen_step}_root${cl_seqlen_root_degree}_vocab_from${cl_vocab_start}_to${cl_vocab_end}_step${cl_vocab_step}_root${cl_vocab_root_degree}.json"
if [[ $zero_stage -gt 0 ]]; then
sed "s/CONFIG_BATCH_SIZE/${global_batch_size}/" ${template_json} \
    | sed "s/CONFIG_MBSIZE/${batch_size}/" \
    | sed "s/LOG_INTERVAL/${log_interval}/" \
    | sed "s/ZERO_STAGE/${zero_stage}/" \
    | sed "s/PRESCALE_GRAD/false/" \
    | sed "s/DATA_EFFICIENCY_SEED/${seed}/" \
    | sed "s/DATA_SAMPLING_NUM_WORKERS/${num_workers}/" \
    | sed "s#CONFIG_CL_CLUSTER_PATH#${data_cluster_path}#" \
    | sed "s#CL_SEQLEN_METRIC_NAME#${cl_seqlen_metric}#" \
    | sed "s/CONFIG_CL_SEQLEN_MIN/${cl_seqlen_start}/" \
    | sed "s/CONFIG_CL_SEQLEN_MAX/${seq_len}/" \
    | sed "s/CONFIG_CL_SEQLEN_DURATION/${cl_seqlen_step}/" \
    | sed "s/CONFIG_CL_SEQLEN_ROOT_DEGREE/${cl_seqlen_root_degree}/" \
    | sed "s#CONFIG_CL_VOCAB_SAMPLE_PATH#${index_to_sample_path}#" \
    | sed "s#CONFIG_CL_VOCAB_METRIC_PATH#${index_to_metric_path}#" \
    | sed "s/CONFIG_CL_VOCAB_MIN/${cl_vocab_start}/" \
    | sed "s/CONFIG_CL_VOCAB_MAX/${cl_vocab_end}/" \
    | sed "s/CONFIG_CL_VOCAB_DURATION/${cl_vocab_step}/" \
    | sed "s/CONFIG_CL_VOCAB_ROOT_DEGREE/${cl_vocab_root_degree}/" \
      > ${config_json}
else
sed "s/CONFIG_BATCH_SIZE/${global_batch_size}/" ${template_json} \
    | sed "s/CONFIG_MBSIZE/${batch_size}/" \
    | sed "s/LOG_INTERVAL/${log_interval}/" \
    | sed "s/ZERO_STAGE/${zero_stage}/" \
    | sed "s/PRESCALE_GRAD/true/" \
    | sed "s/DATA_EFFICIENCY_SEED/${seed}/" \
    | sed "s/DATA_SAMPLING_NUM_WORKERS/${num_workers}/" \
    | sed "s#CONFIG_CL_CLUSTER_PATH#${data_cluster_path}#" \
    | sed "s#CL_SEQLEN_METRIC_NAME#${cl_seqlen_metric}#" \
    | sed "s/CONFIG_CL_SEQLEN_MIN/${cl_seqlen_start}/" \
    | sed "s/CONFIG_CL_SEQLEN_MAX/${seq_len}/" \
    | sed "s/CONFIG_CL_SEQLEN_DURATION/${cl_seqlen_step}/" \
    | sed "s/CONFIG_CL_SEQLEN_ROOT_DEGREE/${cl_seqlen_root_degree}/" \
    | sed "s#CONFIG_CL_VOCAB_SAMPLE_PATH#${index_to_sample_path}#" \
    | sed "s#CONFIG_CL_VOCAB_METRIC_PATH#${index_to_metric_path}#" \
    | sed "s/CONFIG_CL_VOCAB_MIN/${cl_vocab_start}/" \
    | sed "s/CONFIG_CL_VOCAB_MAX/${cl_vocab_end}/" \
    | sed "s/CONFIG_CL_VOCAB_DURATION/${cl_vocab_step}/" \
    | sed "s/CONFIG_CL_VOCAB_ROOT_DEGREE/${cl_vocab_root_degree}/" \
      > ${config_json}
fi

deepspeed_options=" \
    --deepspeed \
    --deepspeed_config ${config_json} \
    --zero-stage ${zero_stage} \
    --pipeline-model-parallel-size ${pp_size}"

if [[ "${no_pp}" = "true" ]]; then
deepspeed_options="${deepspeed_options} \
    --no-pipeline-parallel"
fi

if [ "${activation_checkpoint}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
    --deepspeed-activation-checkpointing"
fi

## When saving checkpoint to a storage with cache, their could be consistency
## issue of the pointer to latest checkpoint. Here we find the correct pointer
## and broadcast it to all nodes.
iteration_file="$checkpoint_path/latest_checkpointed_iteration.txt"
iteration_file_2="$checkpoint_path/latest"
iteration=0
for (( node = 0; node <= num_node-1; node++ ))
do
    if $(ssh -q worker-"$node" "test -f \"$iteration_file\""); then
        local_iteration=$(ssh -q worker-"$node" cat $iteration_file)
        iteration=$(( ${local_iteration} > ${iteration} ? ${local_iteration} :  ${iteration} ))
    fi
done
if [[ $iteration -gt 0 ]]; then
    iteration_2="global_step${iteration}"
    ds_ssh "echo $iteration > $iteration_file"
    ds_ssh "echo $iteration_2 > $iteration_file_2"
fi

deepspeed ${dir}/../../../../pretrain_gpt.py ${megatron_options} ${data_options} ${deepspeed_options} &>> ${log_path}/${jobname}_${host}_${current_time}.log