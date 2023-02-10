FROM mcr.microsoft.com/azureml/curated/acpt-pytorch-1.11-py38-cuda11.5-gpu

RUN pip install wandb
RUN pip install deepspeed==0.8.0
    

