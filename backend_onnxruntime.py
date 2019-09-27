import onnxruntime
import time
import backend
import numpy as np

import utils


class BackendOnnxruntime(backend.Backend):
    def __init__(self):
        super(BackendOnnxruntime, self).__init__()
        self.session = None

    def name(self):
        return "onnxruntime"

    def version(self):
        return onnxruntime.__version__

    def load(self, model, enable_profiling=False, batch_size=1):
        utils.debug("running on {}".format(onnxruntime.get_device()))
        utils.debug("model path = {}".format(model.path))
        self.model = model
        self.enable_profiling = enable_profiling
        #https://microsoft.github.io/onnxruntime/auto_examples/plot_profiling.html
        options = onnxruntime.SessionOptions()
        if enable_profiling:
            options.enable_profiling = True
        if utils.DEBUG:
            options.session_log_severity_level = 0


        options.session_thread_pool_size=2
        options.enable_sequential_execution=True
        options.set_graph_optimization_level(3)
        self.session = onnxruntime.InferenceSession(model.path, options)
        self.inputs = [meta.name for meta in self.session.get_inputs()]
        self.outputs = [meta.name for meta in self.session.get_outputs()]
        utils.debug("inputs of onnxruntime is {}".format(self.inputs))
        utils.debug("outputs of onnxruntime is {}".format(self.outputs))

    def __del__(self):
        if self.enable_profiling:
            prof_file = self.session.end_profiling()
            print("profile file = {}".format(prof_file))

    def forward_once(self, img):
        run_options = onnxruntime.RunOptions()
        if utils.DEBUG:
            run_options.run_log_severity_level = 0
        start = time.time()
        result = self.session.run(self.outputs, {self.inputs[0]: img}, run_options=run_options)
        end = time.time()  # stop timer
        return end - start

    def forward(self, img, warmup=True, num_warmup=100, num_iterations=100):
        utils.debug("image shape = {}".format(np.shape(img)))
        if warmup:
            for ii in range(num_warmup,):
                self.forward_once(img)
        res = []
        for i in range(num_iterations):
            t = self.forward_once(img)
            utils.debug("processing iteration = {} which took {}".format(i, t))
            res.append(t)
        return res
