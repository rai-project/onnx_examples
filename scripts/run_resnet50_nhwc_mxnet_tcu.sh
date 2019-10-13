#!/bin/bash

trap "exit" INT

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

export MXNET_CUDA_ALLOW_TENSOR_CORE=1
export MXNET_CUDA_TENSOR_OP_MATH_ALLOW_CONVERSION=1
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0

declare -a batch_sizes=(
    1
    2
    4
    8
    16
    32
    # 64
    # 128
    # 256
    # 512
    # 1024
)

NUM_WARMUP=1
NUM_ITERATIONS=1

HOST_NAME=$(hostname)
GPU_NAME=$(nvidia-smi --query-gpu="name" --format=csv | sed -n 2p | tr -s ' ' | tr ' ' '_')
RESULTS_DIR=results/mxnet_tcu_nhwc/${GPU_NAME}

mkdir -p ${RESULTS_DIR}
nvidia-smi -x -q -a >${RESULTS_DIR}/nvidia_smi.xml

for BATCH_SIZE in "${batch_sizes[@]}"; do

    MODEL_IDX=16
    BATCH_SIZE_OPT=--batch_size=${BATCH_SIZE}

    echo "Running MXNET batchsize=${BATCH_SIZE}"

    python tcu/run_mxnet_nhwc.py ${BATCH_SIZE_OPT} --num_warmup=$NUM_WARMUP --num_iterations=$NUM_ITERATIONS --model_idx=$MODEL_IDX
done
