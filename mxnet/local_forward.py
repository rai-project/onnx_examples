# run local models given a path, default to './mxnet_models/'

import os
import argparse
import time
import mxnet as mx
import numpy as np

file_path = os.path.realpath(__file__)
dir_name = os.path.dirname(file_path)

os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"


def xprint(s):
    pass


parser = argparse.ArgumentParser(
    description='Predict ImageNet classes from a given image')
parser.add_argument('--model_name', type=str, required=False, default='resnet50_v1',
                    help='name of the model to use')
parser.add_argument('--batch_size', type=int, required=False, default=1,
                    help='batch size to use')
parser.add_argument('--input_dim', type=int, required=False, default=224,
                    help='input dimension')
parser.add_argument('--input_channels', type=int, required=False, default=3,
                    help='input channels')
parser.add_argument('--num_iterations', type=int, required=False, default=30,
                    help='number of iterations to run')
parser.add_argument('--num_warmup', type=int, required=False, default=5,
                    help='number of warmup iterations to run')
parser.add_argument('--model_idx', type=int, required=False, default=2,
                    help='model idx')
opt = parser.parse_args()

model_name = opt.model_name
batch_size = opt.batch_size
input_dim = opt.input_dim
input_channels = opt.input_channels
num_iterations = opt.num_iterations
num_warmup = opt.num_warmup
model_idx = opt.model_idx

ctx = mx.gpu() if len(mx.test_utils.list_gpus()) else mx.cpu()

sym, arg_params, aux_params = mx.model.load_checkpoint(
    dir_name + '/mxnet_models/'+model_name, 0)

data_names = [
    graph_input
    for graph_input in sym.list_inputs()
    if graph_input not in arg_params and graph_input not in aux_params
]

net = mx.mod.Module(
    symbol=sym,
    data_names=[data_names[0]],
    context=ctx,
    label_names=None,
)

input_shape = (batch_size, input_channels, input_dim, input_dim)

img = mx.random.uniform(
    shape=input_shape, ctx=ctx)

net.bind(for_training=False, data_shapes=[
         (data_names[0], input_shape)], label_shapes=net._label_shapes)

net.set_params(arg_params, aux_params, allow_missing=True)


def forward_once():
    mx.nd.waitall()
    start = time.time()
    prob = net.predict(img)
    mx.nd.waitall()
    end = time.time()  # stop timer
    return end - start


for i in range(num_warmup):
    forward_once()

res = []
for i in range(num_iterations):
    t = forward_once()
    res.append(t)

res = np.multiply(res, 1000)

print("{},{},{},{},{},{}".format(model_idx+1, model_name, batch_size, np.min(t),
                                 np.average(t), np.max(t)))
