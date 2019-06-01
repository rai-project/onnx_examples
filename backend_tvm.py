# Not complete. see
# https://docs.tvm.ai/tutorials/frontend/from_onnx.html
# https://github.com/shinh/opbench/blob/0a5d7a92cace9361f206866e2705bc98c9e07973/drivers/tvm.py

import nnvm
import nnvm.compiler
import onnx
import tvm

import time
from collections import namedtuple

import backend


class BackendTVM(backend.Backend):
    def __init__(self):
        super(BackendTVM, self).__init__()
        self.session = None
        self.ctx = tvm.gpu() if len(tvm.test_utils.list_gpus()) else tvm.cpu()

    def name(self):
        return "TVM"

    def version(self):
        return TVM.__version__

    def load(self, model):
        onnx_model = onnx.load_model(model.path)
        symbol, params = nnvm.frontend.from_onnx(onnx_model)
        input_names = symbol.list_input_names()

        shape_dict = {}
        dtype_dict = {}
        for name, value in zip(input_names, inputs):
            shape_dict[name] = value.shape
            dtype_dict[name] = value.dtype
        for name, value in params.items():
            shape_dict[name] = value.shape
            dtype_dict[name] = value.dtype
        with nnvm.compiler.build_config(opt_level=3):
            graph, lib, params = nnvm.compiler.build(
                symbol, target, shape=shape_dict, dtype=dtype_dict, params=params
            )
        tvm_inputs = []
        for input in inputs:
            tvm_inputs.append(tvm.nd.array(input, ctx=ctx))
        tvm_outputs = []
        for output in sample_outputs:
            tvm_outputs.append(tvm.nd.empty(output.shape, output.dtype, ctx=ctx))
        graph_module = graph_runtime.create(graph, lib, ctx)

        self.input_names = input_names
        self.params = {k: tvm.nd.array(v, ctx=ctx) for k, v in params.items()}
        self.tvm_inputs = tvm_inputs
        self.tvm_outputs = tvm_outputs
        self.graph_module = graph_module

    def forward_once(self, img):
        start = time.time()
        result = self.model.forward(img)
        end = time.time()  # stop timer
        return end - start

    def forward(self, img, warmup=True):
        Batch = namedtuple("Batch", ["data"])
        img = mx.nd.array(img, ctx=self.ctx)
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

