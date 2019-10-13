#!/bin/bash

trap "exit" INT

export MXNET_CUDA_ALLOW_TENSOR_CORE=0
export TF_DISABLE_CUDNN_TENSOR_OP_MATH=0
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0

declare -a batch_sizes=(
	1
	2
	4
	8
	16
	32
)

mkdir -p results/renset50/mxnet
mkdir -p results/renset50/mxnet_fp16

for BATCH_SIZE in "${batch_sizes[@]}"; do
	OUTPUTFILE=results/renset50/mxnet/batchsize_${BATCH_SIZE}.csv
	FP16OUTPUTFILE=results/renset50/mxnet_fp16/batchsize_${BATCH_SIZE}.csv

	echo "Running MXNET batchsize=${BATCH_SIZE}"
	rm -fr ${OUTPUTFILE}
	rm -fr ${FP16OUTPUTFILE}
    MXNET_CUDA_ALLOW_TENSOR_CORE=1 MXNET_CUDA_TENSOR_OP_MATH_ALLOW_CONVERSION=1 python main.py --dtype="float16" --batch_size=${BATCH_SIZE} --backend=mxnet --num_warmup=5 --num_iterations=30 --model_idx=16 --debug >>${FP16OUTPUTFILE}
    python main.py --dtype="float32" --batch_size=${BATCH_SIZE} --backend=mxnet --num_warmup=5 --num_iterations=30 --model_idx=16 --debug >>${OUTPUTFILE}
done
