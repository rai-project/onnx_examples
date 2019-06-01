import onnx
import warnings

warnings.filterwarnings("ignore")

import os
import utils

# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from onnx_tf.backend import prepare
from onnx_tf.common import supports_device
import tensorflow as tf
import time

import backend


class BackendTensorflow(backend.Backend):
    def __init__(self):
        super(BackendTensorflow, self).__init__()
        self.session = None
        self.device = "/device:GPU:0" if supports_device("CUDA") else "/cpu:0"
        utils.debug("running on {}".format(self.device))

    def name(self):
        return "tensorflow"

    def version(self):
        return tf.__version__

    def load(self, model):
        utils.debug("loading onnx model {} from disk".format(model.path))
        self.onnx_model = onnx.load(model.path)
        utils.debug("loaded onnx model")
        with tf.device(self.device):
            self.model = prepare(self.onnx_model)
        utils.debug("prepared onnx model")
        self.session = tf.Session(
            graph=tf.import_graph_def(
                self.model.predict_net.graph.as_graph_def(), name=""
            )
        )
        self.inputs = self.session.graph.get_tensor_by_name(
            self.model.predict_net.tensor_dict[
                self.model.predict_net.external_input[0]
            ].name
        )
        self.outputs = self.session.graph.get_tensor_by_name(
            self.model.predict_net.tensor_dict[
                self.model.predict_net.external_output[0]
            ].name
        )
        utils.debug("loaded onnx model")

    def forward_once(self, img):
        start = time.time()
        result = self.session.run(self.output, {self.inputs: img})
        end = time.time()  # stop timer
        return end - start

    def forward(self, img, warmup=True):
        if warmup:
            self.forward_once(img)
        return self.forward_once(img)
