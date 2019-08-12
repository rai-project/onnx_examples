#!/bin/bash


export MXNET_CUDA_ALLOW_TENSOR_CORE=0
export TF_DISABLE_CUDNN_TENSOR_OP_MATH=0
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0


BATCH_SIZE=256
OUTPUTFILE=results/mxnet_${BATCH_SIZE}.csv

BATCH_SIZE_OPT=--batch_size=${BATCH_SIZE}

rm -fr ${OUTPUTFILE}
for i in `seq 0 29`
do
   echo "infer using model $i"
   if [[ "$i" -eq 0 ]];then # arcface
       python main.py ${BATCH_SIZE_OPT} --debug --backend=mxnet --model_idx=$i --input_dim=112 >> ${OUTPUTFILE}
   elif [[ "$i" -eq 7 ]];then # emotion_ferplus
      python main.py ${BATCH_SIZE_OPT} --debug --backend=mxnet --model_idx=$i --input_dim=64 >> ${OUTPUTFILE}
   elif [[ "$i" -eq 10 ]];then # mnist
      python main.py ${BATCH_SIZE_OPT} --debug --backend=mxnet --model_idx=$i --input_dim=28 >> ${OUTPUTFILE}
   elif [[ "$i" -eq 24 ]];then # tiny_yolo
      python main.py ${BATCH_SIZE_OPT} --debug --backend=mxnet --model_idx=$i --input_dim=416 >> ${OUTPUTFILE}
   else
      python main.py ${BATCH_SIZE_OPT} --debug --backend=mxnet --model_idx=$i >> ${OUTPUTFILE}
   fi
done
