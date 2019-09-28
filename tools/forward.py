import argparse
import time
import numpy as np
import mxnet as mx
from cuda_utils import DeviceReset, cudaDeviceSynchronize
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

parser = argparse.ArgumentParser(
    description='Predict ImageNet classes from a given image')
parser.add_argument('--model', type=str, required=True,
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

# cuda_profiler_start()

# Load Model
model_name = opt.model
batch_size = opt.batch_size
input_dim = opt.input_dim
pretrained = True
ctx = mx.gpu()
net = get_model(model_name, pretrained=pretrained, ctx=ctx)

# 224x224

net.hybridize(static_alloc=True, static_shape=True)

net.collect_params().reset_ctx(ctx)


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


# forward_once()
# forward_once()
forward_once()

t = 0
t=forward_once()
t=forward_once()
t=forward_once()
t=forward_once()
with nvtx.profile_range('forward'):
     t = forward_once()
# cuda_profiler_stop()

print(t*1000)

del net

cuda_profiler_stop()

cudaDeviceSynchronize()

del ctx

# DeviceReset(0)

# tmpdir = "/tmp/models/{}/".format(model_name)
# net.export(tmpdir + "model")
