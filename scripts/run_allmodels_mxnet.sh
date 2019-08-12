#!/bin/bash


export MXNET_CUDA_ALLOW_TENSOR_CORE=0
export TF_DISABLE_CUDNN_TENSOR_OP_MATH=0
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0


BATCH_SIZE=256
OUTPUTFILE=$1_${BATCH_SIZE}.csv

BATCH_SIZE_OPT=--batch_size=${BATCH_SIZE}

for i in `seq 0 29`
do
   echo "infer using model $i"
   if [[ "$i" -eq 0 ]];then # arcface
       python main.py ${BATCH_SIZE_OPT} --debug --backend=mxnet --model_idx=$i --input_dim=112 >> output
   elif [[ "$i" -eq 7 ]];then # emotion_ferplus
      python main.py ${BATCH_SIZE_OPT} --debug --backend=mxnet --model_idx=$i --input_dim=64 >> output
   elif [[ "$i" -eq 10 ]];then # mnist
      python main.py ${BATCH_SIZE_OPT} --debug --backend=mxnet --model_idx=$i --input_dim=28 >> output
   elif [[ "$i" -eq 24 ]];then # tiny_yolo
      python main.py ${BATCH_SIZE_OPT} --debug --backend=mxnet --model_idx=$i --input_dim=416 >> output
   else
      python main.py ${BATCH_SIZE_OPT} --debug --backend=mxnet --model_idx=$i >> output
   fi
done
