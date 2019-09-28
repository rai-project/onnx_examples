from PIL import Image
import mxnet as mx
from mxnet.contrib import onnx as onnx_mxnet
import time
from collections import namedtuple
import utils
import numpy as np
from mxnet import profiler
from extra.mxnet_shufflenet import ShuffleNet
import gluoncv
from mxnet import gluon
from image_net_labels import labels
import backend
# from cuda_profiler import cuda_profiler_start, cuda_profiler_stop

# see https://github.com/awslabs/deeplearning-benchmark/blob/master/onnx_benchmark/import_benchmarkscript.py


class BackendMXNet(backend.Backend):
    def __init__(self):
        super(BackendMXNet, self).__init__()
        self.is_run = False
        self.session = None
        self.model_info = None
        self.ctx = mx.gpu() if len(mx.test_utils.list_gpus()) else mx.cpu()
        self.enable_profiling = False

    def name(self):
        return "mxnet"

    def version(self):
        return mx.__version__

    def load(self, model, enable_profiling=False):
        self.model_info = model
        self.enable_profiling = enable_profiling

        # print(model.path)
        # print(model.name)

        self.sym, self.arg, self.aux = onnx_mxnet.import_model(model.path)

        if model.name == "Emotion-FerPlus":
            # download from https://github.com/awslabs/mxnet-model-server/blob/master/docs/model_zoo.md
            params_path = "/home/ubuntu/test/FERPlus-0000.params"
            symbol_path = "/home/ubuntu/test/FERPlus-symbol.json"

            self.sym, self.arg, self.aux = mx.model.load_checkpoint(
                "/home/ubuntu/test/FERPlus", 0)
        model_metadata = onnx_mxnet.get_model_metadata(model.path)
        self.data_names = [
            graph_input
            for graph_input in self.sym.list_inputs()
            if graph_input not in self.arg and graph_input not in self.aux
        ]

        self.graph_outputs = self.sym.list_outputs()

        self.model = gluon.nn.SymbolBlock(
            outputs=self.sym, inputs=mx.sym.var(self.data_names[0], dtype='float32'))
        net_params = self.model.collect_params()
        for param in self.arg:
            if param in net_params:
                net_params[param]._load_init(self.arg[param], ctx=self.ctx)
        for param in self.aux:
            if param in net_params:
                net_params[param]._load_init(self.aux[param], ctx=self.ctx)

        if model.name == "Shufflenet":
            # download from https://github.com/RoGoSo/shufflenet-gluon/blob/master/model.py
            self.model = ShuffleNet()
            self.model.initialize(ctx=self.ctx)

        self.model.hybridize(static_alloc=True, static_shape=True)
        # mx.visualization.plot_network( self.sym,  node_attrs={"shape": "oval", "fixedsize": "false"})

        if enable_profiling:
            profiler.set_config(
                profile_all=True,
                # profile_symbolic=True,
                # profile_imperative=True,
                # profile_api=True,
                filename=model.name + "_profile.json",
                continuous_dump=True,
            )  # Stats printed by dumps() call

    def run_batch(net, data):
        results = []
        for batch in data:
            outputs = net(batch)
            results.extend([o for o in outputs.asnumpy()])
        return np.array(results)

    def forward_once(self, input, validate=False):
        if self.is_run:
            mx.nd.waitall()
            self.is_run = True
        start = time.time()
        prob = self.model.forward(input)
        mx.nd.waitall()
        end = time.time()  # stop timer
        if validate:
            prob = prob.asnumpy()
            prob = np.squeeze(prob)
            a = np.argsort(prob)[::-1]
            for i in a[0:5]:
                print("probability=%f, class=%s" % (prob[i], labels[i]))
        return end - start

    def transform(self, img):
        return np.expand_dims(img, axis=0).astype(np.float32)

    def forward(self, img, warmup=True, num_warmup=100, num_iterations=100, validate=False):
        img = mx.nd.array(img, ctx=self.ctx, dtype="float32")
        utils.debug("image_shape={}".format(np.shape(img)))
        # utils.debug("datanames={}".format(self.data_names))
        # utils.debug("datashapes={}".format(data_shapes))
        # utils.debug("img_shape={}".format(img.shape))
        # print(img)
        utils.debug("num_warmup = {}".format(num_warmup))
        if warmup:
            for i in range(num_warmup):
                self.forward_once(img)
        res = []
        if self.enable_profiling:
            profiler.set_state("run")
        # cuda_profiler_start()
        for i in range(num_iterations):
            t = self.forward_once(img, validate=validate)
            # utils.debug("processing iteration = {} which took {}".format(i, t))
            res.append(t)
        # cuda_profiler_stop()
        if self.enable_profiling:
            mx.nd.waitall()
            profiler.set_state("stop")
            profiler.dump(finished=False)
            # print(profiler.dumps())
        return res
