#!/bin/sh

export MXNET_CUDA_ALLOW_TENSOR_CORE=0
export TF_DISABLE_CUDNN_TENSOR_OP_MATH=0
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0


for i in {0..29}
do
   echo "infer using model $i"
   python main.py --debug --backend=mxnet --model_idx=$i >> output
done
