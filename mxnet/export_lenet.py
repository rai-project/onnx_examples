# pylint: skip-file
from __future__ import print_function

import mxnet as mx
from mxnet import autograd
from mxnet import init
from mxnet.gluon import nn
from mxnet import gluon
import mxnet.ndarray as nd
import time
import numpy as np
import argparse


# Parse CLI arguments

parser = argparse.ArgumentParser(description='MXNet Gluon LeNet Example')
parser.add_argument('--batch-size', type=int, default=100,
                    help='batch size for training and testing (default: 100)')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate (default: 0.1)')
opt = parser.parse_args()


# define network

batch_size = opt.batch_size
epoch_nums = opt.epochs
lr = opt.lr

net = nn.HybridSequential()
with net.name_scope():
    net.add(
        nn.Conv2D(channels=6, kernel_size=5, strides=1,
                  padding=2, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=5, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Flatten(),
        nn.Dense(120, activation='relu'),
        nn.Dense(84, activation='relu'),
        nn.Dense(10),
    )


def try_gpu():
    try:
        ctx = mx.gpu()
        _ = nd.zeros((1,), ctx=ctx)
    except mx.base.MXNetError:
        ctx = mx.cpu()
    return ctx


def transformer(data, label):
    return nd.transpose(data.astype(np.float32), (2, 0, 1)).asnumpy() / 255, label.astype(np.int32)


train_set = gluon.data.vision.MNIST(
    train=True,
    transform=transformer,
)

train_loader = gluon.data.DataLoader(
    dataset=train_set,
    batch_size=batch_size,
    shuffle=True
)

test_set = gluon.data.vision.MNIST(
    train=False,
    transform=transformer,
)

test_loader = gluon.data.DataLoader(
    dataset=test_set,
    batch_size=batch_size,
    shuffle=False
)

ctx = try_gpu()

net.initialize(ctx=ctx, init=init.Xavier())

criterion = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(
    params=net.collect_params(),
    optimizer='sgd',
    optimizer_params={'learning_rate': lr, 'momentum': 0.9},
)

for epoch in range(epoch_nums):
    loss_sum = 0
    start_time = time.time()
    for X, y in train_loader:
        X, y = X.as_in_context(ctx), y.as_in_context(ctx)
        with autograd.record():
            output = net(X)
            loss = criterion(output, y)
        loss.backward()
        trainer.step(batch_size)
        loss_sum += loss.mean().asscalar()

    test_acc = nd.array([0.0], ctx=ctx)
    test_acc = 0
    total = 0
    for X, y in test_loader:
        X, y = X.as_in_context(ctx), y.as_in_context(ctx)
        output = net(X)
        predicted = output.argmax(axis=1)
        test_acc += (predicted == y.astype(np.float32)).sum()
        total += y.size
    print('epoch: %d, train loss: %.03f, test acc: %.03f, time %.1f sec' % (epoch + 1,
                                                                            loss_sum / len(train_loader), test_acc.asscalar() / total, time.time() - start_time))


# net.save_parameters('mnist-0000.params')
net.hybridize()
net.forward(mx.nd.random.uniform(
    shape=[1, 1, 28, 28], ctx=ctx))
net.export('lenet')
