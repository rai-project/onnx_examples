#!/bin/bash

trap "exit" INT

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

export caffe2_CUDA_ALLOW_TENSOR_CORE=0
export TF_DISABLE_CUBLAS_TENSOR_OP_MATH=0
export caffe2_CUBLAS_AUTOTUNE_DEFAULT=0

declare -a batch_sizes=(
    1
)

NUM_WARMUP=1
NUM_ITERATIONS=1

HOST_NAME=$(hostname)
GPU_NAME=$(nvidia-smi --query-gpu="name" --format=csv | sed -n 2p | tr -s ' ' | tr ' ' '_')
RESULTS_DIR=profile_results/caffe2/cublas_log/${GPU_NAME}

mkdir -p ${RESULTS_DIR}
nvidia-smi -x -q -a >${RESULTS_DIR}/nvidia_smi.xml

for BATCH_SIZE in "${batch_sizes[@]}"; do

    BATCH_SIZE_OPT=--batch_size=${BATCH_SIZE}

    echo "Running caffe2 batchsize=${BATCH_SIZE}"

    for i in $(seq 0 29); do
        echo "infer using model $i"
        CUBLAS_LOGDEST_DBG_PATH="${RESULTS_DIR}/$((i + 1))_${BATCH_SIZE}.log"
        echo "${CUBLAS_LOGDEST_DBG_PATH}"
        rm -f ${CUBLAS_LOGDEST_DBG_PATH}

        # run onnx models
        if [[ "$i" -eq 0 ]]; then # arcface
            CUBLAS_LOGINFO_DBG=1 CUBLAS_LOGDEST_DBG=${CUBLAS_LOGDEST_DBG_PATH} python main.py ${BATCH_SIZE_OPT} --backend=caffe2 --short_output --num_warmup=$NUM_WARMUP --num_iterations=$NUM_ITERATIONS --model_idx=$i --input_dim=112
        elif [[ "$i" -eq 7 ]]; then # emotion_ferplus
            CUBLAS_LOGINFO_DBG=1 CUBLAS_LOGDEST_DBG=${CUBLAS_LOGDEST_DBG_PATH} python main.py ${BATCH_SIZE_OPT} --backend=caffe2 --short_output --num_warmup=$NUM_WARMUP --num_iterations=$NUM_ITERATIONS --model_idx=$i --input_dim=64 --input_channels=1
        elif [[ "$i" -eq 10 ]]; then # mnist
            CUBLAS_LOGINFO_DBG=1 CUBLAS_LOGDEST_DBG=${CUBLAS_LOGDEST_DBG_PATH} python main.py ${BATCH_SIZE_OPT} --backend=caffe2 --short_output --num_warmup=$NUM_WARMUP --num_iterations=$NUM_ITERATIONS --model_idx=$i --input_dim=28 --input_channels=1
        elif [[ "$i" -eq 24 ]]; then # tiny_yolo
            CUBLAS_LOGINFO_DBG=1 CUBLAS_LOGDEST_DBG=${CUBLAS_LOGDEST_DBG_PATH} python main.py ${BATCH_SIZE_OPT} --backend=caffe2 --short_output --num_warmup=$NUM_WARMUP --num_iterations=$NUM_ITERATIONS --model_idx=$i --input_dim=416
        else
            CUBLAS_LOGINFO_DBG=1 CUBLAS_LOGDEST_DBG=${CUBLAS_LOGDEST_DBG_PATH} python main.py ${BATCH_SIZE_OPT} --backend=caffe2 --short_output --num_warmup=$NUM_WARMUP --num_iterations=$NUM_ITERATIONS --model_idx=$i
        fi
    done
done
