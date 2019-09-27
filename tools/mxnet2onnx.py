import numpy as np
import mxnet as mx
from mxnet.contrib import onnx as onnx_mxnet

converted_onnx_filename='alexnet.onnx'

# Export MXNet model to ONNX format via MXNet's export_model API
converted_onnx_filename=onnx_mxnet.export_model('model-symbol.json', 'model-0000.params', [(4, 3,224,224)], np.float32, converted_onnx_filename)

# Check that the newly created model is valid and meets ONNX specification.
import onnx
model_proto = onnx.load(converted_onnx_filename)
onnx.checker.check_model(model_proto)