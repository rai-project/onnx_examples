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
	64
	128
	256
	512
	1024
)

mkdir -p results/mxnet

for BATCH_SIZE in "${batch_sizes[@]}"; do
	OUTPUTFILE=results/mxnet/batchsize_${BATCH_SIZE}.csv
	BATCH_SIZE_OPT=--batch_size=${BATCH_SIZE}

	echo "Running MXNET batchsize=${BATCH_SIZE}"
	rm -fr ${OUTPUTFILE}
	rm -fr ${OUTPUTFILE}.gz
	for i in $(seq 0 29); do
		echo "infer using model $i"
		if [[ "$i" -eq 0 ]]; then # arcface
			python main.py ${BATCH_SIZE_OPT} --backend=mxnet --short_output --num_warmup=1 --num_iterations=1 --model_idx=$i --input_dim=112 >>${OUTPUTFILE}
		elif [[ "$i" -eq 7 ]]; then # emotion_ferplus
			python main.py ${BATCH_SIZE_OPT} --backend=mxnet --short_output --num_warmup=1 --num_iterations=1 --model_idx=$i --input_dim=64 --input_channels=1 >>${OUTPUTFILE}
		elif [[ "$i" -eq 10 ]]; then # mnist
			python main.py ${BATCH_SIZE_OPT} --backend=mxnet --short_output --num_warmup=1 --num_iterations=1 --model_idx=$i --input_dim=28 --input_channels=1 >>${OUTPUTFILE}
		elif [[ "$i" -eq 24 ]]; then # tiny_yolo
			python main.py ${BATCH_SIZE_OPT} --backend=mxnet --short_output --num_warmup=1 --num_iterations=1 --model_idx=$i --input_dim=416 >>${OUTPUTFILE}
		else
			python main.py ${BATCH_SIZE_OPT} --backend=mxnet --short_output --num_warmup=1 --num_iterations=1 --model_idx=$i >>${OUTPUTFILE}
		fi
	done
	gzip ${OUTPUTFILE}
done
