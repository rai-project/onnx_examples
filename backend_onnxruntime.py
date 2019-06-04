import onnxruntime
import time
import backend

import utils


class BackendOnnxruntime(backend.Backend):
    def __init__(self):
        super(BackendOnnxruntime, self).__init__()
        self.session = None

    def name(self):
        return "onnxruntime"

    def version(self):
        return onnxruntime.__version__

    def load(self, model, enable_profiling=False):
        self.model = model
        self.enable_profiling = enable_profiling
        # options = onnxruntime.SessionOptions()
        # if enable_profiling:
        #     options.enable_profiling = True
        options = None
        self.session = onnxruntime.InferenceSession(model.path, options)
        self.inputs = [meta.name for meta in self.session.get_inputs()]
        self.outputs = [meta.name for meta in self.session.get_outputs()]
        utils.debug("inputs of onnxruntime is {}".format(self.inputs))
        utils.debug("outputs of onnxruntime is {}".format(self.outputs))

    # def __del__(self):
    #     if self.enable_profiling:
    #         prof_file = self.session.end_profiling()
    #         print("profile file = {}".format(prof_file))

    def forward_once(self, img):
        start = time.time()
        result = self.session.run(self.outputs, {self.inputs[0]: img})
        end = time.time()  # stop timer
        return end - start

    def forward(self, img, warmup=True):
        if warmup:
            self.forward_once(img)
        return self.forward_once(img)
