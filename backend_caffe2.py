import torch
import onnx
import time
import caffe2.python.onnx.backend

import backend


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

    def load(self, model, enable_profiling=False):
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
        self.enable_profiling = enable_profiling
        if enable_profiling:
            self.session.net.AddObserver("ProfileObserver")
            self.session.net.AddObserver("TimeObserver")

    def __del__(self):
        if self.enable_profiling:
            profile_observer = self.session.net.GetObserver("ProfileObserver")
            profile_observer.dump()

    def forward_once(self, img):
        start = time.time()
        result = self.session.run(img)
        end = time.time()  # stop timer
        return end - start

    def forward(self, img, warmup=True):
        if warmup:
            self.forward_once(img)
        return self.forward_once(img)
