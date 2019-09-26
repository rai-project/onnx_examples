import os
import click
import sys
import copy
import traceback
import onnx

DEBUG = False
VERBOSE = False
QUIET = False


def debug(msg):
    if DEBUG and not QUIET:
        click.echo(click.style("[DEBU] " + msg, fg="green"), err=True)


def error(msg):
    if not QUIET:
        click.echo(click.style("[ERRO] " + msg, fg="red"), err=True)


def warn(msg):
    if not QUIET:
        click.echo(click.style("[WARN] " + msg, fg="yellow"), err=True)


def halt(msg):
    error(msg)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)


def require(pred, msg=None):
    if not pred:
        if not QUIET:
            if msg:
                click.echo(click.style(
                    "[INTERNAL ERROR] " + msg, fg="red"), err=True)
            else:
                click.echo(click.style("[INTERNAL ERROR]", fg="red"), err=True)
    assert pred


def change_input_dim(model):
    # Use some symbolic name not used for any other dimension
    sym_batch_dim = "N"
    # or an actal value
    actual_batch_dim = 4

    # The following code changes the first dimension of every input to be batch-dim
    # Modify as appropriate ... note that this requires all inputs to
    # have the same batch_dim
    inputs = model.graph.input
    for input in inputs:
        if len(input.type.tensor_type.shape.dim) == 0:
            continue
        # Checks omitted.This assumes that all inputs are tensors and have a shape with first dim.
        # Add checks as needed.
        dim1 = input.type.tensor_type.shape.dim[0]
        # update dim to be a symbolic value
        dim1.dim_param = sym_batch_dim
        # or update it to be an actual value:
        # dim1.dim_value = actual_batch_dim


def apply(transform, infile, outfile):
    model = onnx.load(infile)
    transform(model)
    onnx.save(model, outfile)


def fix_batch_size(model):
    res_path = os.path.join(os.path.dirname(model.path), 'model_batch.onnx')
    apply(change_input_dim, model.path, res_path)
    model.path = res_path
    return model


def fix_batch_size0(model):
    import os
    import onnx
    from onnx import numpy_helper
    from onnx import shape_inference

    if os.path.basename(model.path) == "model_batch.onxx":
        return model

    # load model to edit
    mp = onnx.load(model.path)
    input_name = 'data'
    batch_symbol = 'batch'
    print(mp.graph.input)
    graph_input = [i for i in mp.graph.input if i.name == input_name][0]
    # change input to have symbolic batch
    graph_input.type.tensor_type.shape.dim[0].dim_param = batch_symbol
    # clear stale shape inference
    mp.graph.ClearField('value_info')
    # fix the reshape op that outputs OC2_DUMMY_0 to forward 'batch' using 0
    shape_const = [
        i for i in mp.graph.initializer if i.name == 'OC2_DUMMY_1'][0]
    shape_const.CopyFrom(numpy_helper.from_array(
        np.asarray([0, 2048], dtype=np.int64), 'OC2_DUMMY_1'))
    # add batch to graph output shape
    mp.graph.output[0].type.tensor_type.shape.dim[0].dim_param = batch_symbol
    # run shape inference
    mp = shape_inference.infer_shapes(mp)
    # save model with batch
    model.path = os.path.join(os.path.dirname(model.path), 'model_batch.onnx')
    onnx.save(mp, model.path)
    return model
