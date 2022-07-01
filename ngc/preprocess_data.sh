#!/bin/bash

IMAGE=nvcr.io/nvidia/pytorch:22.03-py3
MEGATRON=/home/nvidia/Megatron-DeepSpeed
NAME=preprocess

INPUT=/data/converted/oscar.json
VOCAB=${MEGATRON}/vocab/zh_word.vocab
KEYS=text
DATA_PREFIX=/data/upload/oscar/word_oscar
WORKERS=16

EXE=${MEGATRON}/tools/preprocess_data.py   # For Chinese

docker run --rm --gpus all \
        --name=${NAME} \
        -v /data/:/data/ -v ${HOME}:${HOME} \
        --privileged \
        --shm-size=20g \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -it ${IMAGE} \
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