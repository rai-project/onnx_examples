# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
# pylint: disable= arguments-differ,unused-argument,missing-docstring,too-many-lines
"""ResNets, implemented in Gluon."""
from __future__ import division

__all__ = ['ResNetV1',
           'BasicBlockV1',
           'BottleneckV1',
           'resnet50_v1',
           'get_resnet']

from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
import mxnet as mx
from mxnet.gluon.nn import BatchNorm

# Helpers
def _conv3x3(channels, stride, in_channels):
    return nn.Conv2D(channels, kernel_size=3, strides=stride, padding=1,
                     use_bias=False, in_channels=in_channels, layout="NHWC")


# Blocks
class BasicBlockV1(HybridBlock):
    r"""BasicBlock V1 from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    This is used for ResNet V1 for 18, 34 layers.

    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    last_gamma : bool, default False
        Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
    use_se : bool, default False
        Whether to use Squeeze-and-Excitation module
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    def __init__(self, channels, stride, downsample=False, in_channels=0,
                 last_gamma=False, use_se=False, norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(BasicBlockV1, self).__init__(**kwargs)
        self.body = nn.HybridSequential(prefix='')
        self.body.add(_conv3x3(channels, stride, in_channels))
        self.body.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        self.body.add(nn.Activation('relu'))
        self.body.add(_conv3x3(channels, 1, channels))
        if not last_gamma:
            self.body.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        else:
            self.body.add(norm_layer(gamma_initializer='zeros',
                                     **({} if norm_kwargs is None else norm_kwargs)))

        self.se = None

        if downsample:
            self.downsample = nn.HybridSequential(prefix='')
            self.downsample.add(nn.Conv2D(channels, kernel_size=1, strides=stride,
                                          use_bias=False, in_channels=in_channels))
            self.downsample.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x

        x = self.body(x)

        if self.se:
            w = F.contrib.AdaptiveAvgPooling2D(x, output_size=1, layout="NHWC")
            w = self.se(w)
            x = F.broadcast_mul(x, w.expand_dims(axis=2).expand_dims(axis=2))

        if self.downsample:
            residual = self.downsample(residual)

        x = F.Activation(residual+x, act_type='relu')

        return x


class BottleneckV1(HybridBlock):
    r"""Bottleneck V1 from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    This is used for ResNet V1 for 50, 101, 152 layers.

    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    last_gamma : bool, default False
        Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
    use_se : bool, default False
        Whether to use Squeeze-and-Excitation module
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    def __init__(self, channels, stride, downsample=False, in_channels=0,
                 last_gamma=False, use_se=False, norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(BottleneckV1, self).__init__(**kwargs)
        self.body = nn.HybridSequential(prefix='')
        self.body.add(nn.Conv2D(channels//4, kernel_size=1, strides=stride, layout="NHWC"))
        self.body.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        self.body.add(nn.Activation('relu'))
        self.body.add(_conv3x3(channels//4, 1, channels//4))
        self.body.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        self.body.add(nn.Activation('relu'))
        self.body.add(nn.Conv2D(channels, kernel_size=1, strides=1, layout="NHWC"))

        self.se = None

        if not last_gamma:
            self.body.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        else:
            self.body.add(norm_layer(gamma_initializer='zeros',
                                     **({} if norm_kwargs is None else norm_kwargs)))

        if downsample:
            self.downsample = nn.HybridSequential(prefix='')
            self.downsample.add(nn.Conv2D(channels, kernel_size=1, strides=stride,
                                          use_bias=False, in_channels=in_channels, layout="NHWC"))
            self.downsample.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x

        x = self.body(x)

        if self.se:
            w = F.contrib.AdaptiveAvgPooling2D(x, output_size=1, layout="NHWC")
            w = self.se(w)
            x = F.broadcast_mul(x, w.expand_dims(axis=2).expand_dims(axis=2))

        if self.downsample:
            residual = self.downsample(residual)

        x = F.Activation(x + residual, act_type='relu')
        return x




# Nets
class ResNetV1(HybridBlock):
    r"""ResNet V1 model from
    `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    block : HybridBlock
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    channels : list of int
        Numbers of channels in each block. Length should be one larger than layers list.
    classes : int, default 1000
        Number of classification classes.
    thumbnail : bool, default False
        Enable thumbnail.
    last_gamma : bool, default False
        Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
    use_se : bool, default False
        Whether to use Squeeze-and-Excitation module
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    def __init__(self, block, layers, channels, classes=1000, thumbnail=False,
                 last_gamma=False, use_se=False, norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(ResNetV1, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            if thumbnail:
                self.features.add(_conv3x3(channels[0], 1, 0))
            else:
                self.features.add(nn.Conv2D(channels[0], 7, 2, 3, use_bias=False, layout="NHWC"))
                self.features.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
                self.features.add(nn.Activation('relu'))
                self.features.add(nn.MaxPool2D(3, 2, 1, layout="NHWC"))

            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                self.features.add(self._make_layer(block, num_layer, channels[i+1],
                                                   stride, i+1, in_channels=channels[i],
                                                   last_gamma=last_gamma, use_se=use_se,
                                                   norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            self.features.add(nn.GlobalAvgPool2D(layout="NHWC"))

            self.output = nn.Dense(classes, in_units=channels[-1], dtype="float16")

    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0,
                    last_gamma=False, use_se=False, norm_layer=BatchNorm, norm_kwargs=None):
        layer = nn.HybridSequential(prefix='stage%d_'%stage_index)
        with layer.name_scope():
            layer.add(block(channels, stride, channels != in_channels, in_channels=in_channels,
                            last_gamma=last_gamma, use_se=use_se, prefix='',
                            norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            for _ in range(layers-1):
                layer.add(block(channels, 1, False, in_channels=channels,
                                last_gamma=last_gamma, use_se=use_se, prefix='',
                                norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        return layer

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)

        return x




# Specification
resnet_spec = {18: ('basic_block', [2, 2, 2, 2], [64, 64, 128, 256, 512]),
               34: ('basic_block', [3, 4, 6, 3], [64, 64, 128, 256, 512]),
               50: ('bottle_neck', [3, 4, 6, 3], [64, 256, 512, 1024, 2048]),
               101: ('bottle_neck', [3, 4, 23, 3], [64, 256, 512, 1024, 2048]),
               152: ('bottle_neck', [3, 8, 36, 3], [64, 256, 512, 1024, 2048])}

resnet_net_versions = [ResNetV1 ]
resnet_block_versions = [{'basic_block': BasicBlockV1, 'bottle_neck': BottleneckV1}]


# Constructor
def get_resnet(version, num_layers, pretrained=False, ctx=cpu(),
               root='~/.mxnet/models', use_se=False, **kwargs):
    r"""ResNet V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    ResNet V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    version : int
        Version of ResNet. Options are 1, 2.
    num_layers : int
        Numbers of layers. Options are 18, 34, 50, 101, 152.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    use_se : bool, default False
        Whether to use Squeeze-and-Excitation module
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    assert num_layers in resnet_spec, \
        "Invalid number of layers: %d. Options are %s"%(
            num_layers, str(resnet_spec.keys()))
    block_type, layers, channels = resnet_spec[num_layers]
    assert 1 <= version <= 2, \
        "Invalid resnet version: %d. Options are 1 and 2."%version
    resnet_class = resnet_net_versions[version-1]
    block_class = resnet_block_versions[version-1][block_type]
    net = resnet_class(block_class, layers, channels, **kwargs)
    return net

def resnet50_v1(**kwargs):
    r"""ResNet-50 V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    return get_resnet(1, 50, use_se=False, norm_kwargs={'axis':3}, **kwargs)

