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
    m = {DeviceType.CPU: caffe2_pb2.CPU,
         DeviceType.CUDA: workspace.GpuDeviceType}
    return core.DeviceOption(m[device.type], device.device_id)


class BackendCaffe2(backend.Backend):
    def __init__(self):
        super(BackendCaffe2, self).__init__()
        self.session = None
        self.input_data = None
        self.input_name = None
        self.device = "CUDA:0" if torch.cuda.is_available() else "CPU"
        self.enable_profiling = False
        self.profile_observer = None
        self.init_net = None
        self.predict_net = None

    def name(self):
        return "caffe2"

    def version(self):
        return torch.__version__

    def load(self, model, enable_profiling=False, cuda_profile=False, batch_size=1):
        onnx_model_proto = ModelProto()
        with open(model.path, "rb") as onnx_model:
            onnx_model_proto.ParseFromString(onnx_model.read())

        self.model = onnx_model_proto
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

        device_option = get_device_option(Device(self.device))

        init_net, predict_net = c2.onnx_graph_to_caffe2_net(
            onnx_model_proto, device=self.device
        )

        ws = Workspace()
        device_option = get_device_option(Device(self.device))

        initialized = {init.name for init in self.model.graph.initializer}

        for tp in self.model.graph.initializer:
            ws.FeedBlob(tp.name, onnx.numpy_helper.to_array(tp), device_option)

        for value_info in self.model.graph.input:
            if value_info.name in initialized:
                continue
            shape = list(
                d.dim_value for d in value_info.type.tensor_type.shape.dim)
            ws.FeedBlob(
                value_info.name,
                np.ones(
                    shape,
                    dtype=onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[
                        value_info.type.tensor_type.elem_type
                    ],
                ),
                device_option,
            )

        self.uninitialized = [
            value_info.name for value_info in self.model.graph.input if value_info.name not in initialized]

        ws.CreateNet(init_net)
        ws.RunNet(init_net.name)

        ws.CreateNet(predict_net)

        # predict_net = core.Net(predict_net)
        enable_profiling = False
        if enable_profiling:
            print(type(predict_net))
            self.profile_observer = predict_net.AddObserver(
                "ProfileObserver"
            )
            predict_net.AddObserver("TimeObserver")

        self.session = ws
        self.model = model
        self.enable_profiling = enable_profiling
        self.init_net = init_net
        self.predict_net = predict_net

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
        self.session = caffe2.python.onnx.backend.prepare(
            self.model, self.device)
        self.enable_profiling = enable_profiling

    def __del__(self):
        if self.enable_profiling and self.profile_observer is not None:
            print("dassa")
            self.profile_observer.dump()

    def forward_once(self, img):
        start = time.time()
        # self.session.FeedBlob(self.uninitialized[0], img)
        # result = self.session.RunNet(self.predict_net.name)
        result = self.session.run(img)
        end = time.time()  # stop timer
        return end - start

    def forward(self, img, warmup=True, num_warmup=100, num_iterations=100, validate=False):
        if warmup:
            for i in range(num_warmup):
                self.forward_once(img)
        res = []
        for i in range(num_iterations):
            cuda_profiler_start()
            res.append(self.forward_once(img))
            cuda_profiler_stop()
        return res
