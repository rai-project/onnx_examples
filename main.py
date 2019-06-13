import argparse
import json
import os.path
import sys
import click
import glob
import traceback
import numpy as np


import info
import utils
import input_image
from models import models



def get_backend(backend):
    if backend == "tensorflow" or backend == "tf":
        from backend_tf import BackendTensorflow

        backend = BackendTensorflow()
    elif backend == "caffe2":
        from backend_caffe2 import BackendCaffe2

        backend = BackendCaffe2()
    elif backend == "onnxruntime":
        from backend_onnxruntime import BackendOnnxruntime

        backend = BackendOnnxruntime()
    elif backend == "null":
        from backend_null import BackendNull

        backend = BackendNull()
    elif backend == "pytorch":
        from backend_pytorch import BackendPytorch

        backend = BackendPytorch()
    elif backend == "pytorch-native":
        from backend_pytorch_native import BackendPytorchNative

        backend = BackendPytorchNative()
    elif backend == "mxnet":
        from backend_mxnet import BackendMXNet

        backend = BackendMXNet()
    elif backend == "tflite":
        from backend_tflite import BackendTflite

        backend = BackendTflite()
    else:
        raise ValueError("unknown backend: " + backend)
    utils.debug("Loading {} backend version {}".format(backend.name(), backend.version()))
    return backend


# @click.option(
#     "-d",
#     "--debug",
#     type=click.BOOL,
#     is_flag=True,
#     help="print debug messages to stderr.",
#     default=False,
# )
# @click.option(
#     "-q",
#     "--quiet",
#     type=click.BOOL,
#     is_flag=True,
#     help="don't print messages",
#     default=False,
# )


@click.command()
@click.option("--backend", type=click.STRING, default="onnxruntime")
@click.option("--batch_size", type=click.INT, default=1)
@click.option("--input_dim", type=click.INT, default=224)
@click.option("--model_idx", type=click.INT, default=0)
@click.option("--profile/--no-profile", help="don't perform layer-wise profiling", default=False)
@click.option(
    "--debug/--no-debug", help="print debug messages to stderr.", default=False
)
@click.option("--quiet/--no-quiet", help="don't print messages", default=False)
@click.pass_context
@click.version_option()
def main(ctx, backend, batch_size, input_dim, model_idx, profile, debug, quiet):
    utils.DEBUG = debug
    utils.QUIET = quiet

    model = models[model_idx]

    if model.path is None:
        raise Exception("unable to find model in {}".format(model.name))

    utils.debug("Using {} model".format(model.name))

    try:
        backend = get_backend(backend)
    except Exception as err:
        traceback.print_exc()
        sys.exit(1)

    img = input_image.get(model, input_dim, batch_size)

    try:
        backend.load(model, enable_profiling=profile)
    except Exception as err:
        traceback.print_exc()
        sys.exit(1)

    try:
        t = backend.forward(img, num_warmup=2, num_iterations=10)
    except Exception as err:
        traceback.print_exc()
        sys.exit(1)


    print("elapsed time = {}ms".format(np.average(t) * 1000))



if __name__ == "__main__":
    main()
