#!/bin/bash

pyenv virtualenv miniconda3-4.3.30 dlperf
pyenv activate dlperf

pip install onnx onnxmltools
pip install future click numba
pip install onnxruntime-gpu
pip install mxnet-cu101mkl
pip install gluoncv

mc cp -r s3/store.carml.org/dlperf/mxnet_models ./mxnet

mc cp -r s3/store.carml.org/dlperf/onnx_models/ ~/data/carml/dlperf/
