#!/bin/bash

mc cp -r s3/store.carml.org/dlperf/mxnet_models ./mxnet

mc cp -r s3/store.carml.org/dlperf/onnx_models/* ~/data/carml/dlperf/
