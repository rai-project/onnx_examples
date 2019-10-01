#!/bin/bash

trap "exit" INT

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

export caffe2_CUDA_ALLOW_TENSOR_CORE=0
export TF_DISABLE_CUDNN_TENSOR_OP_MATH=0
export caffe2_CUDNN_AUTOTUNE_DEFAULT=0

declare -a batch_sizes=(
	1
)

NUM_WARMUP=5
NUM_ITERATIONS=30

HOST_NAME=$(hostname)
GPU_NAME=$(nvidia-smi --query-gpu="name" --format=csv | sed -n 2p | tr -s ' ' | tr ' ' '_')
RESULTS_DIR=${DIR}/../results/caffe2/${GPU_NAME}

mkdir -p ${RESULTS_DIR}
nvidia-smi -x -q -a >${RESULTS_DIR}/nvidia_smi.xml

for BATCH_SIZE in "${batch_sizes[@]}"; do
	OUTPUTFILE=${RESULTS_DIR}/batchsize_${BATCH_SIZE}.csv
	BATCH_SIZE_OPT=--batch_size=${BATCH_SIZE}

	rm -fr ${OUTPUTFILE}

	echo "Running caffe2 batchsize=${BATCH_SIZE}"
	rm -fr ${OUTPUTFILE}
	for i in $(seq 0 29); do
		echo "Running model=${i}"
		# run onnx models
		if [[ "$i" -eq 0 ]]; then # arcface
			python main.py ${BATCH_SIZE_OPT} --backend=caffe2 --short_output --num_warmup=$NUM_WARMUP --num_iterations=$NUM_ITERATIONS --model_idx=$i --input_dim=112 >>${OUTPUTFILE}
		elif [[ "$i" -eq 6 ]]; then # duc
			python main.py ${BATCH_SIZE_OPT} --backend=caffe2 --short_output --num_warmup=$NUM_WARMUP --num_iterations=$NUM_ITERATIONS --model_idx=$i --input_dim=800 >>${OUTPUTFILE}
		elif [[ "$i" -eq 7 ]]; then # emotion_ferplus
			python main.py ${BATCH_SIZE_OPT} --backend=caffe2 --short_output --num_warmup=$NUM_WARMUP --num_iterations=$NUM_ITERATIONS --model_idx=$i --input_dim=64 --input_channels=1 >>${OUTPUTFILE}
		elif [[ "$i" -eq 10 ]]; then # mnist
			python main.py ${BATCH_SIZE_OPT} --backend=caffe2 --short_output --num_warmup=$NUM_WARMUP --num_iterations=$NUM_ITERATIONS --model_idx=$i --input_dim=28 --input_channels=1 >>${OUTPUTFILE}
		elif [[ "$i" -eq 24 ]]; then # tiny_yolo
			python main.py ${BATCH_SIZE_OPT} --backend=caffe2 --short_output --num_warmup=$NUM_WARMUP --num_iterations=$NUM_ITERATIONS --model_idx=$i --input_dim=416 >>${OUTPUTFILE}
		else
			python main.py ${BATCH_SIZE_OPT} --backend=caffe2 --short_output --num_warmup=$NUM_WARMUP --num_iterations=$NUM_ITERATIONS --model_idx=$i >>${OUTPUTFILE}
		fi
	done
	gzip ${OUTPUTFILE}
done
