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
RESULTS_DIR=profile_results/mxnet_tcu/nsight/${GPU_NAME}

mkdir -p ${RESULTS_DIR}
nvidia-smi -x -q -a >${RESULTS_DIR}/nvidia_smi.xml



NSYS=/opt/nvidia/nsight-systems/2019.5.1/bin/nsys

for BATCH_SIZE in "${batch_sizes[@]}"; do

    MODEL_IDX=16
    BATCH_SIZE_OPT=--batch_size=${BATCH_SIZE}

    echo "Running MXNET batchsize=${BATCH_SIZE}"

    echo "infer using model $i"
    NSIGHT_PATH="${RESULTS_DIR}/$((MODEL_IDX + 1))_${BATCH_SIZE}.log"
    echo "${CUDNN_LOGDEST_DBG_PATH}"
    rm -f ${CUDNN_LOGDEST_DBG_PATH}

    ${NSYS} profile --force-overwrite=true --trace=cuda,cudnn,cublas,nvtx --sample=none --output=${NSIGHT_PATH} python tcu/run_mxnet.py ${BATCH_SIZE_OPT} --num_warmup=$NUM_WARMUP --num_iterations=$NUM_ITERATIONS --model_idx=$MODEL_IDX
    # ${NSYS} profile --force-overwrite=true --trace=cuda,cudnn,cublas,nvtx --sample=none --output=${NSIGHT_PATH} --export=sqlite  python tcu/run_mxnet.py ${BATCH_SIZE_OPT} --num_warmup=$NUM_WARMUP --num_iterations=$NUM_ITERATIONS --model_idx=$MODEL_IDX

    gzip ${NSIGHT_PATH}.qdrep
    gzip ${NSIGHT_PATH}.sqlite
done
