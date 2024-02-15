import os
import sys
from setuptools import setup, find_packages

with open("README.md") as fh:
    long_description = fh.read()

setup(
    name='zbpp_light',
    python_requires='>=3.8',
    version='0.0.1',
    packages=find_packages(),
    description=(
        'Zero Bubble Pipeline Parallelism (ZBPP) light weight implementation. For more details refer to https://github.com/sail-sg/zero-bubble-pipeline-parallelism.'
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=('Qi Penghui, Wan Xinyi, Huang Guangxing, Lin Min'),
    author_email='wanxy@sea.com',
    url='https://github.com/sail-sg/zero-bubble-pipeline-parallelism',    
)