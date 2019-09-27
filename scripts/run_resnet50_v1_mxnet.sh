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

for BATCH_SIZE in "${batch_sizes[@]}"; do
	OUTPUTFILE=results/renset50/mxnet/batchsize_${BATCH_SIZE}.csv
	BATCH_SIZE_OPT=--batch_size=${BATCH_SIZE}

	echo "Running MXNET batchsize=${BATCH_SIZE}"
	rm -fr ${OUTPUTFILE}
	rm -fr ${OUTPUTFILE}.gz
    python main.py ${BATCH_SIZE_OPT} --backend=mxnet --num_warmup=5 --num_iterations=30 --model_idx=16 >>${OUTPUTFILE}
	gzip ${OUTPUTFILE}
done
