#!/bin/bash

IMAGE=pytorch2203
MEGATRON=/home/nvidia/Megatron-DeepSpeed

INPUT=/data/converted/oscar.json

VOCAB=${MEGATRON}/vocab/zh_word.vocab

KEYS=text
DATA_PREFIX=/data/upload/oscar/word_oscar

WORKERS=16


# EXE=tools/zh/preprocess_data_zh.py   # For Chinese
EXE=${MEGATRON}/tools/preprocess_data.py   # For Chinese
docker exec ${IMAGE} bash -c "pip install jieba; cd ${MEGATRON}; mkdir /data/upload/oscar ; \
python ${EXE} \
       --input '${INPUT}' \
       --output-prefix ${DATA_PREFIX} \
       --vocab ${VOCAB} \
       --json-keys ${KEYS} \
       --dataset-impl mmap \
       --workers ${WORKERS} \
       --tokenizer-type ZHBertTokenizer \
       --append-eod
       "