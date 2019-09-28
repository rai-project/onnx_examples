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
import nvtxpy as nvtx

from gluoncv.model_zoo import get_model
from gluoncv.data.transforms.presets.imagenet import transform_eval

os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"


parser = argparse.ArgumentParser(
    description='Predict ImageNet classes from a given image')
parser.add_argument('--model', type=str, required=False, default='resnet50_v1',
                    help='name of the model to use')
parser.add_argument('--batch_size', type=int, required=False, default=1,
                    help='batch size to use')
parser.add_argument('--input_dim', type=int, required=False, default=224,
                    help='input dimension')
opt = parser.parse_args()


class cuda_profiler_start():
    import numba.cuda as cuda
    cuda.profile_start()


class cuda_profiler_stop():
    import numba.cuda as cuda
    cuda.profile_stop()


# Load Model
model_name = opt.model
batch_size = opt.batch_size
input_dim = opt.input_dim
pretrained = True
ctx = mx.gpu() if len(mx.test_utils.list_gpus()) else mx.cpu()
net = get_model(model_name, pretrained=pretrained, ctx=ctx)

net.hybridize(static_alloc=True, static_shape=True)


def forward_once():
    input = mx.random.uniform(
        shape=(batch_size, 3, input_dim, input_dim), ctx=ctx)
    mx.nd.waitall()
    start = time.time()
    prob = net.forward(input)
    prob = prob.asnumpy()
    mx.nd.waitall()
    end = time.time()  # stop timer
    return end - start


print(mx.__version__)

forward_once()
# forward_once()
# forward_once()

t = 0
with nvtx.profile_range('forward'):
    t = forward_once()

print(t*1000)