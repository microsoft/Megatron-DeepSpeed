FROM mcr.microsoft.com/azureml/curated/acpt-pytorch-1.11-py38-cuda11.5-gpu


USER root:root

RUN pip install pybind11
RUN pip

RUN pip install git+https://github.com/microsoft/DeepSpeed.git

# add a100-topo.xml
RUN mkdir -p /opt/microsoft/
RUN wget -O /opt/microsoft/a100-topo.xml https://hpcbenchmarks.blob.core.windows.net/bookcorpus/data/a100-topo.xml

# to use on A100, enable env var below in your job
ENV NCCL_TOPO_FILE="/opt/microsoft/a100-topo.xml"
