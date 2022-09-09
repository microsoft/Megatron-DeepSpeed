FROM ptebic.azurecr.io/public/azureml/aifx/stable-ubuntu2004-cu113-py38-torch1110:latest

USER root:root

RUN pip install pybind11 regex

RUN pip install git+https://github.com/microsoft/DeepSpeed.git

#RUN git clone https://github.com/prathikr/DeepSpeed.git&&\
#	cd DeepSpeed &&\
#	git checkout prathikrao/detach-gradient &&\
#	pip install -e .
