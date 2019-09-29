## Experiments of inference using onnx models

### Models

We look at 30 ONNX models from the ONNX model zoo. Some models do not support bach size more than 1 when running MXNet or other backends.
To run the models in MXNet with batch size > 1, we use equivalent MXNet models form other sources.

|   ID | Model Name         | The original ONNX model supports batch size > 1? | The source mode to run in MXNet | Notes                                                                                                                           |
| ---: | ------------------ | ------------------------------------------------ | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
|    0 | ArcFace            | yes                                              | ONNX                            |                                                                                                                                 |
|    1 | BVLC_AlexNet       | no                                               | GluonCV                         |                                                                                                                                 |
|    2 | BVLC_CaffeNet      | no                                               | Caffe                           |                                                                                                                                 |
|    3 | BVLC_GoogleNet     | no                                               | Caffe                           |                                                                                                                                 |
|    4 | BVLC_RCNN_ILSVRC13 | no                                               | Caffe                           |                                                                                                                                 |
|    5 | DenseNet-121       | yes                                              | ONNX                            |                                                                                                                                 |
|    6 | DUC                | yes                                              | ONNX                            |                                                                                                                                 |
|    7 | Emotion-FerPlus    | no                                               | None                            | The original ONNX model is converted from CNTK                                                                                  |
|    8 | Inception-v1       | no                                               | MXNet Model Server              |                                                                                                                                 |
|    9 | Inception-v2       | no                                               | None                            | The original ONNX model is converted from Caffe2                                                                                |
|   10 | MNIST              | no                                               | ONNX                            | LeNet. The original ONNX model is converted from CNTK, which does not run. We trained a MXNet LeNet and converted it into ONNX. |
|   11 | MobileNet-v2       | yes                                              | ONNX                            |                                                                                                                                 |
|   12 | ResNet018-v1       | yes                                              | ONNX                            |                                                                                                                                 |
|   13 | ResNet018-v2       | yes                                              | ONNX                            |                                                                                                                                 |
|   14 | ResNet034-v1       | yes                                              | ONNX                            |                                                                                                                                 |
|   15 | ResNet034-v2       | yes                                              | ONNX                            |                                                                                                                                 |
|   16 | ResNet050-v1       | yes                                              | ONNX                            |                                                                                                                                 |
|   17 | ResNet050-v2       | yes                                              | ONNX                            |                                                                                                                                 |
|   18 | ResNet101-v1       | yes                                              | ONNX                            |                                                                                                                                 |
|   19 | ResNet101-v2       | yes                                              | ONNX                            |                                                                                                                                 |
|   20 | ResNet152-v1       | yes                                              | ONNX                            |                                                                                                                                 |
|   21 | ResNet152-v2       | yes                                              | ONNX                            |                                                                                                                                 |
|   22 | Shufflenet         | yes                                              | ONNX                            |                                                                                                                                 |
|   23 | Squeezenet-v1.1    | yes                                              | ONNX                            |                                                                                                                                 |
|   24 | Tiny_YOLO-v2       | yes                                              | ONNX                            |                                                                                                                                 |
|   25 | VGG16-BN           | yes                                              | ONNX                            |                                                                                                                                 |
|   26 | VGG16              | yes                                              | ONNX                            |                                                                                                                                 |
|   27 | VGG19-BN           | yes                                              | ONNX                            |                                                                                                                                 |
|   28 | VGG19              | yes                                              | ONNX                            |                                                                                                                                 |
|   29 | Zfnet512           | no                                               | GluonCV2                        | The original ONNX model is converted from Caffe2                                                                                |

### Install Requirements

1. GPU

```
pyenv virtualenv miniconda3-4.3.30 dlperf
pyenv activate dlperf

pip install onnx onnxmltools
pip install future click numba
pip install onnxruntime-gpu
pip install mxnet-cu101mkl
pip install gluoncv
pip install tensorflow-gpu
pip install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp37-cp37m-linux_x86_64.whl
pip install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp37-cp37m-linux_x86_64.whl
```

2. CPU

Use [`pipenv`](https://github.com/pypa/pipenv) to launch a shell

```
pyenv virtualenv miniconda3-4.3.30 dlperf
pyenv activate dlperf
```

Then install the packages using pip

```
pip install onnx gluoncv mxnet onnxmltools onnxruntime torchvision torch tensorflow onnx-tf future tvm numba click pycodestyle
```

### Setup the Environment

Run

```
./setup_en.sh
```

which set up the following environment variables.

- Disable Autotune

```
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export TF_CUDNN_USE_AUTOTUNE=0
```

- Disable TensorCores

```
export MXNET_CUDA_ALLOW_TENSOR_CORE=0
export TF_DISABLE_CUDNN_TENSOR_OP_MATH=0
```

- Disable bulk mode in MXNet

```
export MXNET_EXEC_BULK_EXEC_INFERENCE=0
export MXNET_EXEC_BULK_EXEC_TRAIN=0
```

### Run Models

1.  Run ONNX models with various backends

```
python main.py --debug --backend=mxnet
python main.py --debug --backend=onnxruntime
python main.py --debug --backend=caffe2
```
Some models only support batch size = 1, see [Models](#Models).

2. Run MXNet models from GluonCV

```
python mxnet/gluon_forward.py --num_warmup=1 --num_iterations=1 --model_name=alexnet --model_idx=1 --batch_size=1
```

3.  Run MXNet models from XXX-symbo.json and XXX-0000.params

```
python mxnet/local_forward.py --num_warmup=1 --num_iterations=1 --model_name=bvlc_caffenet --model_idx=1 --batch_size=1
```

4. Run experiments with scripts

Run all MXNet models with 

```
./scripts/run_allmodels_mxnet.sh
```

### Profiling 

1.  Run with cudnn logging

```
CUDNN_LOGINFO_DBG=1 CUDNN_LOGDEST_DBG=cudnn.log python main.py --debug --backend=mxnet
```

2. Run with cublas logging

```
CUBLAS_LOGINFO_DBG=1 CUBLAS_LOGDEST_DBG=cublas.logpython main.py --debug --backend=mxnet
```


3.  Run with full logging

```
CUBLAS_LOGINFO_DBG=1 CUBLAS_LOGDEST_DBG=cublas.log CUDNN_LOGINFO_DBG=1 CUDNN_LOGDEST_DBG=cudnn.log python main.py --debug --backend=mxnet
```

4. Profile using nvprof

```
nvprof --profile-from-start off --export-profile profiler_output.nvvp -f --print-summary python main.py --backend=mxnet  --num_warmup=1 --num_iterations=1 --model_idx=1
```

5.  Profile using Nsight

```
nsys profile --trace=cuda,cudnn,cublas python main.py --backend=mxnet --num_warmup=1 --num_iterations=1 --model_idx=1
```

### TensorRT

1.  Download TensorRT 5.1.x.x for Ubuntu 18.04 and CUDA 10.1 tar package from https://developer.nvidia.com/nvidia-tensorrt-download.

        wget https://developer.download.nvidia.com/compute/machine-learning/tensorrt/secure/5.1/ga/tars/TensorRT-5.1.5.0.Ubuntu-14.04.5.x86_64-gnu.cuda-10.1.cudnn7.5.tar.gz?fJfT5Un1lcVLX6aTm89YH629UhBMhoyMnb8HdlyVBZ88L5hrC7wwuzkb6sO63qnAY7daItOQus4c3W26kXBA_lx85AUPzImocwEUruEBu03qDyHSUoVqCHBY5C46WL9tOfug-qGNSJ4b-9Jc2aE48YQkymPsgH3AU9twHL8ghhlzzw3aUqZhRh98aUi6kydjT_nMvjt8IImTL8Juhk3mmb_SHMW8mW8xlrs7RhfVKdTw70MRhMtRrQ

2.  Extract archive

         tar xzvf TensorRT-5.1.x.x.<os>.<arch>-gnu.cuda-x.x.cudnn7.x.tar.gz

3.  Install uff python package using pip

         cd TensorRT-5.1.x.x/python
         pip install tensorrt-5.1.x.x-cp3x-none-linux_x86_64.whl
