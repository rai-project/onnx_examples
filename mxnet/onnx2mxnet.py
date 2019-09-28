# convert a onnx model to a mxnet model
import onnx
import numpy as np
import mxnet as mx
from mxnet.contrib import onnx as onnx_mxnet

onnx_filename = 'zfnet512/model.onnx'
converted_mxnet_filename = 'mxnet_models/zfnet512'
input_dim = 224
input_channels = 3

# Import the ONNX model into MXNet's symbolic interface
sym, arg_params, aux_params = onnx_mxnet.import_model(onnx_filename)

ctx = mx.gpu(0)

data_names = [
    graph_input
    for graph_input in sym.list_inputs()
    if graph_input not in arg_params and graph_input not in aux_params
]


net = mx.mod.Module(
    symbol=sym,
    data_names=data_names,
    context=ctx,
    label_names=None,
)

input_shape = (1, input_channels, input_dim, input_dim)

net.bind(for_training=False, data_shapes=[
         (data_names[0], input_shape)], label_shapes=net._label_shapes)
net.set_params(arg_params, aux_params, allow_missing=True)

net.save_checkpoint(converted_mxnet_filename, 0)
