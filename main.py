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
from models import get_models


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
    utils.debug("Loading {} backend version {}".format(
        backend.name(), backend.version()))
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
@click.option("--num_warmup", type=click.INT, default=2)
@click.option("--num_iterations", type=click.INT, default=10)
@click.option("--input_dim", type=click.INT, default=224)
@click.option("--input_channels", type=click.INT, default=3)
@click.option("--model_idx", type=click.INT, default=0)
@click.option("--profile/--no-profile", help="don't perform layer-wise profiling", default=False)
@click.option(
    "--debug/--no-debug", help="print debug messages to stderr.", default=False
)
@click.option("--quiet/--no-quiet", help="don't print messages", default=False)
@click.option("--short_output/--no-short_output", help="shorten the output results", default=True)
@click.option("--output/--no-output", help="don't print output results", default=True)
@click.option("--validate/--no-validate", help="don't validate output results", default=False)
@click.pass_context
@click.version_option()
def main(ctx, backend, batch_size, num_warmup, num_iterations, input_dim, input_channels, model_idx, profile, debug, quiet, short_output, output, validate):
    utils.DEBUG = debug
    utils.QUIET = quiet

    models = get_models(batch_size=batch_size)
    model = models[model_idx]

    if model.path is None:
        raise Exception("unable to find model in {}".format(model.name))

    utils.debug("Using {} model".format(model.name))

    try:
        backend = get_backend(backend)
    except Exception as err:
        traceback.print_exc()
        sys.exit(1)

    img = input_image.get(model, input_dim, input_channels,
                          batch_size=batch_size)

    try:
        if backend.name() == "mxnet" and batch_size > 1:
            model = utils.fix_batch_size(model)
        backend.load(model, enable_profiling=profile)
    except Exception as err:
        traceback.print_exc()
        sys.exit(1)

    try:
        t = backend.forward(img, num_warmup=num_warmup,
                            num_iterations=num_iterations,
                            validate=validate)
    except Exception as err:
        traceback.print_exc()
        sys.exit(1)

    t = np.multiply(t, 1000)
    utils.debug("mode idx = {}, model = {} elapsed time = {}ms".format(
        model_idx, model.name, np.average(t)))
    if output and not short_output:
        print("{},{},{},{},{},{},\"{}\"".format(model_idx+1, model.name, batch_size, np.min(t),
                                                np.average(t), np.max(t), ';'.join(str(x) for x in t)))
    elif output:
        print("{},{},{},{},{},{}".format(model_idx+1, model.name, batch_size, np.min(t),
                                         np.average(t), np.max(t)))


if __name__ == "__main__":
    main()
