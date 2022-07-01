#!/bin/bash
MEGATRON=/home/nvidia/Megatron-DeepSpeed
NAME=pytorch2203

INPUT=/data/converted/oscar.json
VOCAB=${MEGATRON}/vocab/zh_word.vocab
KEYS=text
DATA_PREFIX=/data/upload/oscar/word_oscar
WORKERS=16

EXE=${MEGATRON}/tools/preprocess_data.py   # For Chinese

docker exec ${NAME} \
        /bin/bash \
        -c " cd ${MEGATRON}; pip install -r requirements.txt ;  mkdir /data/upload/oscar ; \
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