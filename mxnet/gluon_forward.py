# run gluoncv models

import argparse
import os
import time
import numpy as np
import mxnet as mx
from mxnet import nd, image
from mxnet.gluon.data.vision import transforms
from gluoncv.utils import export_block
# from .nvtx import (profile_range,
#                    profile_range_push,
#                    profile_range_pop,
#                    profile_mark,
#                    profiled,
#                    getstats,
#                    colors)
# import nvtxpy as nvtx

from gluoncv.model_zoo import get_model
from gluoncv.data.transforms.presets.imagenet import transform_eval

os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"


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
parser.add_argument('--model_idx', type=int, required=False, default=1,
                    help='model idx')
opt = parser.parse_args()


class cuda_profiler_start():
    import numba.cuda as cuda
    cuda.profile_start()


class cuda_profiler_stop():
    import numba.cuda as cuda
    cuda.profile_stop()


# Load Model
model_name = opt.model_name
batch_size = opt.batch_size
input_dim = opt.input_dim
input_channels = opt.input_channels
num_iterations = opt.num_iterations
num_warmup = opt.num_warmup
model_idx = opt.model_idx

pretrained = True
ctx = mx.gpu() if len(mx.test_utils.list_gpus()) else mx.cpu()
net = get_model(model_name, pretrained=pretrained, ctx=ctx)

net.hybridize(static_alloc=True, static_shape=True)

input_shape = (batch_size, input_channels, input_dim, input_dim)

input = mx.random.uniform(
    shape=input_shape, ctx=ctx)


def forward_once():
    mx.nd.waitall()
    start = time.time()
    prob = net.forward(input)
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

if model_name == "alexnet":
    model_name = "bvlc_"+model_name

print("{},{},{},{},{},{}".format(model_idx+1, model_name, batch_size, np.min(t),
                                 np.average(t), np.max(t)))
