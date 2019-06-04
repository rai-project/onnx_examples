# See
# https://github.com/shinh/opbench/blob/0a5d7a92cace9361f206866e2705bc98c9e07973/drivers/trt.py
# https://github.com/NERSC/inference_benchmarks/blob/a44d2594ae8daee53165e5f4ae468235ac471002/hep_cnn/onnx/run_tensorrt_onnx.py

from PIL import Image
import time
from collections import namedtuple
import utils
import numpy as np
import tensorrt as trt
from trt.parsers import onnxparser

from image_net_labels import labels
import backend
from cuda_profiler import cuda_profiler_start, cuda_profiler_stop


class BackendTensorRT(backend.Backend):
    def __init__(self):
        super(BackendTensorRT, self).__init__()
        self.session = None
        self.batch_size = 1

    def name(self):
        return "tensorrt"

    def version(self):
        return "unknown"

    def load(self, model_path):
        with builder = trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            with open(model_path, 'rb') as model:
                parser.parse(model.read())
                builder.max_batch_size = batch_size
                # This determines the amount of memory available to the builder when building an optimized engine and should generally be set as high as possible.
                builder.max_workspace_size = 1 << 20
                # When the engine is built, TensorRT makes copies of the weights.
                self.engine = builder.build_cuda_engine(network)

    def forward_once(self):
        # Determine dimensions and create page-locked memory buffers (i.e. won't be swapped to disk) to hold host inputs/outputs.
        h_input = cuda.pagelocked_empty(  # pycuda.driver.pagelocked_empty(shape, dtype, order="C", mem_flags=0)
            self.engine.get_binding_shape(0).volume(), dtype=np.float32)
        h_output = cuda.pagelocked_empty(
            self.engine.get_binding_shape(1).volume(), dtype=np.float32)

        # Allocate device memory for inputs and outputs.
        d_input = cuda.mem_alloc(h_input.nbytes)
        d_output = cuda.mem_alloc(h_output.nbytes)

        # Create a stream in which to copy inputs/outputs and run inference.
        stream = cuda.Stream()

        context = self.engine.create_execution_context()

        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(d_input, h_input, stream)

        start = time.time()
        # Run inference.
        context.execute_async(bindings=[int(d_input), int(
            d_output)], stream_handle=stream.handle)
        end = time.time()

        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(h_output, d_output, stream)

        # Synchronize the stream
        stream.synchronize()

        # Return the inference time.
        return end - start

    def forward(self, warmup=True, num_warmup=100, num_iterations=100):
        if warmup:
            for i in range(num_warmup):
                self.forward_once()
        res = []
        cuda_profiler_start()
        for i in range(num_iterations):
            utils.debug("processing iteration = {}".format(i))
            res.append(self.forward_once())
        cuda_profiler_stop()
        return res
