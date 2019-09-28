import os
import argparse
import time
import mxnet as mx
import numpy as np
from collections import namedtuple

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
opt = parser.parse_args()

model_name = opt.model_name
batch_size = opt.batch_size
input_dim = opt.input_dim

ctx = mx.gpu(0)

with open('synset.txt', 'r') as f:
    labels = [l.rstrip() for l in f]

sym, arg_params, aux_params = mx.model.load_checkpoint(
    'mxnet_models/'+model_name, 0)

net = mx.mod.Module(
    symbol=sym,
    data_names=['data'],
    context=ctx,
    label_names=None,
)

input_shape = (batch_size, 3, input_dim, input_dim)
img = mx.nd.random.uniform(
    shape=input_shape, ctx=ctx)

xprint(net._label_shapes)

net.bind(for_training=False, data_shapes=[
         ('data', input_shape)], label_shapes=net._label_shapes)
net.set_params(arg_params, aux_params, allow_missing=True)

Batch = namedtuple("Batch", ["data"])


def forward_once():
    mx.nd.waitall()
    start = time.time()
    output = net.predict(img)
    prob = output.asnumpy()
    mx.nd.waitall()
    end = time.time()  # stop timer
    return end - start
# prob = np.squeeze(prob)
# a = np.argsort(prob)[::-1]
# for i in a[0:5]:
#     print('probability=%f, class=%s' % (prob[i], labels[i]))


forward_once()
# forward_once()
# forward_once()

t = 0
t = forward_once()

print(t*1000)
