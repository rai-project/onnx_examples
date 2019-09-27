import argparse
import time
import numpy as np
import mxnet as mx
from mxnet import nd, image
from mxnet.gluon.data.vision import transforms
from gluoncv.utils import export_block

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


# Load Model
model_name = opt.model
batch_size = opt.batch_size
input_dim = opt.input_dim
pretrained = True
net = get_model(model_name, pretrained=pretrained)

ctx = mx.context.current_context()
# 224x224
input = mx.random.uniform(shape=(batch_size, 3, input_dim, input_dim), ctx=ctx)

net.hybridize()

start = time.time()
prob = net.forward(input)
mx.nd.waitall()
end = time.time()  # stop timer

prob = prob.asnumpy()
print((end - start)*1000)

# tmpdir = "/tmp/models/{}/".format(model_name)
# net.export(tmpdir + "model")
