FROM ptebic.azurecr.io/public/azureml/aifx/stable-ubuntu2004-cu113-py38-torch1110:latest

USER root:root

RUN pip install pybind11 regex