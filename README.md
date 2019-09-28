## Experiments of inference using onnx models

### Models

|   ID | Model Name         | Source                           | Note                                   |
| ---: | ------------------ | -------------------------------- | -------------------------------------- |
|    0 | ArcFace            | ONNX                             |                                        |
|    1 | BVLC_AlexNet       | GluonCV                          |                                        |
|    2 | BVLC_CaffeNet      | Caffe                            |                                        |
|    3 | BVLC_GoogleNet     | Caffe                            |                                        |
|    4 | BVLC_RCNN_ILSVRC13 | Caffe                            |                                        |
|    5 | DenseNet-121       | ONNX                             |                                        |
|    6 | DUC                | ONNX                             |                                        |
|    7 | Emotion-FerPlus    | Does not work for batch size > 1 | The original ONNX model is from CNTK   |
|    8 | Inception-v1       | MXNet Model Server               |                                        |
|    9 | Inception-v2       | Does not work for batch size > 1 | The original ONNX model is from Caffe2 |
|   10 | MNIST              | MXNet                            | The original ONNX model is from CNTK   |
|   11 | MobileNet-v2       | ONNX                             |                                        |
|   12 | ResNet018-v1       | ONNX                             |                                        |
|   13 | ResNet018-v2       | ONNX                             |                                        |
|   14 | ResNet034-v1       | ONNX                             |                                        |
|   15 | ResNet034-v2       | ONNX                             |                                        |
|   16 | ResNet050-v1       | ONNX                             |                                        |
|   17 | ResNet050-v2       | ONNX                             |                                        |
|   18 | ResNet101-v1       | ONNX                             |                                        |
|   19 | ResNet101-v2       | ONNX                             |                                        |
|   20 | ResNet152-v1       | ONNX                             |                                        |
|   21 | ResNet152-v2       | ONNX                             |                                        |
|   22 | Shufflenet         | ONNX                             |                                        |
|   23 | Squeezenet-v1.1    | ONNX                             |                                        |
|   24 | Tiny_YOLO-v2       | ONNX                             |                                        |
|   25 | VGG16-BN           | ONNX                             |                                        |
|   26 | VGG16              | ONNX                             |                                        |
|   27 | VGG19-BN           | ONNX                             |                                        |
|   28 | VGG19              | ONNX                             |                                        |
|   29 | Zfnet512           | GluonCV2                         | The original ONNX model is from Caffe2 |


### Install Requirements

#### GPU

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

#### CPU

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

### Run with full logging

```
CUBLAS_LOGINFO_DBG=1 CUBLAS_LOGDEST_DBG=cublas.log CUDNN_LOGINFO_DBG=1 CUDNN_LOGDEST_DBG=cudnn.log python main.py --debug --backend=mxnet
```

### Run with cudnn logging

```
CUDNN_LOGINFO_DBG=1 CUDNN_LOGDEST_DBG=cudnn.log python main.py --debug --backend=mxnet
```

### Run with cublas logging

```
CUBLAS_LOGINFO_DBG=1 CUBLAS_LOGDEST_DBG=cublas.logpython main.py --debug --backend=mxnet
```

### Run using nvprof

```
nvprof --profile-from-start off --export-profile profiler_output.nvvp -f --print-summary  python main.py --debug --backend=mxnet
```

#### TensorRT

1.  Download TensorRT 5.1.x.x for Ubuntu 18.04 and CUDA 10.1 tar package from https://developer.nvidia.com/nvidia-tensorrt-download.

        wget https://developer.download.nvidia.com/compute/machine-learning/tensorrt/secure/5.1/ga/tars/TensorRT-5.1.5.0.Ubuntu-14.04.5.x86_64-gnu.cuda-10.1.cudnn7.5.tar.gz?fJfT5Un1lcVLX6aTm89YH629UhBMhoyMnb8HdlyVBZ88L5hrC7wwuzkb6sO63qnAY7daItOQus4c3W26kXBA_lx85AUPzImocwEUruEBu03qDyHSUoVqCHBY5C46WL9tOfug-qGNSJ4b-9Jc2aE48YQkymPsgH3AU9twHL8ghhlzzw3aUqZhRh98aUi6kydjT_nMvjt8IImTL8Juhk3mmb_SHMW8mW8xlrs7RhfVKdTw70MRhMtRrQ

2.  Extract archive

         tar xzvf TensorRT-5.1.x.x.<os>.<arch>-gnu.cuda-x.x.cudnn7.x.tar.gz

3.  Install uff python package using pip

         cd TensorRT-5.1.x.x/python
         pip install tensorrt-5.1.x.x-cp3x-none-linux_x86_64.whl

### Run

```
python main.py --debug --backend=mxnet
python main.py --debug --backend=onnxruntime
python main.py --debug --backend=caffe2
```


### Profile using Nsight

1. ONNX models
   
```
nsys profile --trace=cuda,cudnn,cublas python main.py --backend=mxnet --num_warmup=1 --num_iterations=1 --model_idx=1
```

1. MXNet models
```
nsys profile --trace=cudnn,cublas python forward.py --model vgg16 --batch_size=32
```
