# load a mxnet model export to a onnx model

import onnx
import numpy as np
import mxnet as mx
from mxnet.contrib import onnx as onnx_mxnet

converted_onnx_filename = 'lenet.onnx'

# Export MXNet model to ONNX format via MXNet's export_model API
converted_onnx_filename = onnx_mxnet.export_model(
    'mxnet_models/lenet-symbol.json', 'mxnet_models/lenet-0000.params', [(1, 1, 28, 28)], np.float32, converted_onnx_filename)

# Check that the newly created model is valid and meets ONNX specification.
model_proto = onnx.load(converted_onnx_filename)
onnx.checker.check_model(model_proto)
