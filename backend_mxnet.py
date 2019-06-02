import mxnet as mx
from mxnet.contrib import onnx as onnx_mxnet
import time
from collections import namedtuple
import utils
from mxnet import profiler

from image_net_labels import labels
import backend


class BackendMXNet(backend.Backend):
    def __init__(self):
        super(BackendMXNet, self).__init__()
        self.session = None
        self.ctx = mx.gpu() if len(mx.test_utils.list_gpus()) else mx.cpu()
        self.enable_profiling = False

    def name(self):
        return "mxnet"

    def version(self):
        return mxnet.__version__

    def load(self, model, enable_profiling=False):
        self.sym, self.arg, self.aux = onnx_mxnet.import_model(model.path)
        self.data_names = [
            graph_input
            for graph_input in self.sym.list_inputs()
            if graph_input not in self.arg and graph_input not in self.aux
        ]
        self.model = mx.mod.Module(
            symbol=self.sym,
            data_names=self.data_names,
            context=self.ctx,
            label_names=None,
        )
        if enable_profiling:
            profiler.set_config(profile_all=True,
                                            filename=model.name+'_profile.json',  # File used for chrome://tracing visualization
                                                                    continuous_dump=True,
                                                                                            aggregate_stats=True)  # Stats printed by dumps() call

    def forward_once(self, img, validate=False):
        profiler.set_state('run')
        start = time.time()
        result = self.model.forward(img)
        profiler.pause()
        end = time.time()  # stop timer
        if validate:
            import numpy as np

            prob = self.model.get_outputs()[0].asnumpy()
            prob = np.squeeze(prob)
            a = np.argsort(prob)[::-1]
            for i in a[0:5]:
                print("probability=%f, class=%s" % (prob[i], labels[i]))
        return end - start

    def forward(self, img, warmup=True):
        Batch = namedtuple("Batch", ["data"])
        img = mx.nd.array(img, ctx=self.ctx)
        utils.debug("input shape = {}".format(img.shape))
        self.model.bind(
            for_training=False,
            data_shapes=[(self.data_names[0], img.shape)],
            label_shapes=None,
        )
        self.model.set_params(
            arg_params=self.arg,
            aux_params=self.aux,
            allow_missing=True,
            allow_extra=True,
        )
        img = Batch([img])
        if warmup:
            self.forward_once(img)
        return self.forward_once(img)

