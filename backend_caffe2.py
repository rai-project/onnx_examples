import torch
import onnx
import time
import numpy as np
import caffe2.python.onnx.backend

from cuda_profiler import cuda_profiler_start, cuda_profiler_stop

from onnx import checker, ModelProto
from caffe2.python import core, workspace
from caffe2.python.onnx.workspace import Workspace

from caffe2.python.onnx.backend import Caffe2Backend as c2


from caffe2.proto import caffe2_pb2
from onnx.backend.base import Backend, Device, DeviceType, namedtupledict

from cuda_profiler import cuda_profiler_start, cuda_profiler_stop

import backend


def get_device_option(device):
    m = {DeviceType.CPU: caffe2_pb2.CPU, DeviceType.CUDA: workspace.GpuDeviceType}
    return core.DeviceOption(m[device.type], device.device_id)


class BackendCaffe2(backend.Backend):
    def __init__(self):
        super(BackendCaffe2, self).__init__()
        self.session = None
        self.input_data = None
        self.input_name = None
        self.device = "CUDA:0" if torch.cuda.is_available() else "CPU"

    def name(self):
        return "caffe2"

    def version(self):
        return torch.__version__

    def load(self, model, enable_profiling=False, cuda_profile=False):
        self.model = onnx.load(model.path)
        self.inputs = []
        initializers = set()
        for i in self.model.graph.initializer:
            initializers.add(i.name)
        for i in self.model.graph.input:
            if i.name not in initializers:
                self.inputs.append(i.name)
        self.outputs = []
        for i in self.model.graph.output:
            self.outputs.append(i.name)
        self.session = caffe2.python.onnx.backend.prepare(self.model, self.device)
        print(type(self.session.predict_net))
        self.session.predict_net.type = "async_scheduling"

    def forward_once(self, img):
        start = time.time()
        # self.session.FeedBlob(self.uninitialized[0], img)
        # result = self.session.RunNet(self.predict_net.name)
        result = self.session.run(img)
        end = time.time()  # stop timer
        return end - start

    def forward(
        self, img, warmup=True, num_warmup=100, num_iterations=100, validate=False
    ):
        if warmup:
            for i in range(num_warmup):
                self.forward_once(img)
        res = []
        for i in range(num_iterations):
            res.append(self.forward_once(img))
        return res
