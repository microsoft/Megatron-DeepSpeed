FROM mcr.microsoft.com/azureml/curated/acpt-pytorch-1.11-py38-cuda11.5-gpu

RUN useradd -ms /bin/bash megatron
USER megatron

WORKDIR /home/megatron

COPY --chown=megatron:megatron /Megatron-DeepSpeed /home/megatron/Megatron-DeepSpeed

RUN pip install wandb
RUN pip install deepspeed==0.8.0
RUN cd /home/megatron/Megatron-DeepSpeed/megatron/data && make
    

