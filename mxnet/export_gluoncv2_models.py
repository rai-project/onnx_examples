# https://github.com/osmr/imgclsmob/tree/master/gluon
# pip install gluoncv2

from gluoncv2.model_provider import get_model as glcv2_get_model
import mxnet as mx

ctx = mx.gpu()
net = glcv2_get_model("zfnet512", pretrained=True, ctx=ctx)
net.hybridize()
x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
net(x)
net.export('zfnet512')
