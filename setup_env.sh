export MXNET_CUDA_ALLOW_TENSOR_CORE=0
export TF_DISABLE_CUDNN_TENSOR_OP_MATH=0

export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export TF_CUDNN_USE_AUTOTUNE=0

export MXNET_EXEC_BULK_EXEC_INFERENCE=0
export MXNET_EXEC_BULK_EXEC_TRAIN=0

# clear LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:/home/ubuntu/.gvm/pkgsets/go1.12/global/overlay/lib

export PATH=/opt/nvidia/nsight-systems-cli/2019.5.1/bin:$PATH
