import torch
import torch.onnx as torch_onnx
import time

import backend


class BackendPytorch(backend.Backend):
    def __init__(self):
        super(BackendPytorch, self).__init__()
        self.session = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def name(self):
        return "pytroch"

    def version(self):
        return torch.__version__

    def load(self, model):
        self.model = torch.load(model.path, map_location=lambda storage, loc: storage)
        self.model.eval()

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
        device = "CUDA:0" if torch.cuda.is_available() else "CPU"
        self.model = self.model.to(self.device)

    def forward_once(self, img):
        with torch.no_grad():
            start = time.time()
            result = self.model(img)
            end = time.time()  # stop timer
            return end - start

    def forward(self, img, warmup=True):
        img = torch.tensor(img).float().to(self.device)
        if warmup:
            self.forward_once(img)
        return self.forward_once(img)
