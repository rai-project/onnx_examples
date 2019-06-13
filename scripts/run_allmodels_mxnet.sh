#!/bin/bash

OUTPUTFILE=$1

export MXNET_CUDA_ALLOW_TENSOR_CORE=0
export TF_DISABLE_CUDNN_TENSOR_OP_MATH=0
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0


for i in `seq 0 29`
do
   echo "infer using model $i"
   if [[ "$i" -eq 0 ]];then # arcface
      python main.py --debug --backend=mxnet --model_idx=$i --input_dim=112 >> output
   elif [[ "$i" -eq 7 ]];then # emotion_ferplus
      python main.py --debug --backend=mxnet --model_idx=$i --input_dim=64 >> output
   elif [[ "$i" -eq 10 ]];then # mnist
      python main.py --debug --backend=mxnet --model_idx=$i --input_dim=28 >> output
   elif [[ "$i" -eq 24 ]];then # tiny_yolo
      python main.py --debug --backend=mxnet --model_idx=$i --input_dim=416 >> output
   else 
      python main.py --debug --backend=mxnet --model_idx=$i >> output
   fi
done
