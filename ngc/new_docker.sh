IMAGE=nvcr.io/nvidia/pytorch:22.03-py3
NAME=pytorch2203
docker run --gpus all \
        --name=${NAME} \
        -v /data/:/data/ -v ${HOME}:${HOME} \
        --privileged \
        --shm-size=24g \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -itd ${IMAGE} \
        /bin/bash