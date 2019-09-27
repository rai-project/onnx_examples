## Experiments of inference using onnx models

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
conda activate dlperf
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

1. Profile cuda,cudnn,cublas traces
```
nsys profile --trace=cuda,cudnn,cublas python main.py --backend=mxnet --num_warmup=1 --num_iterations=1 --model_idx=1
```