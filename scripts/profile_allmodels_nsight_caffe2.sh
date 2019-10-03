#!/bin/bash

trap "exit" INT

sudo sh -c 'echo 1 >/proc/sys/kernel/perf_event_paranoid'

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

export caffe2_CUDA_ALLOW_TENSOR_CORE=0
export TF_DISABLE_CUDNN_TENSOR_OP_MATH=0
export caffe2_CUDNN_AUTOTUNE_DEFAULT=0

declare -a batch_sizes=(
    1
)

NUM_WARMUP=1
NUM_ITERATIONS=9

HOST_NAME=$(hostname)
GPU_NAME=$(nvidia-smi --query-gpu="name" --format=csv | sed -n 2p | tr -s ' ' | tr ' ' '_')
RESULTS_DIR=profile_results/caffe2/parallel/nsight/${GPU_NAME}

NSYS=/opt/nvidia/nsight-systems/2019.5.1/bin/nsys

mkdir -p ${RESULTS_DIR}
nvidia-smi -x -q -a >${RESULTS_DIR}/nvidia_smi.xml

for BATCH_SIZE in "${batch_sizes[@]}"; do

    BATCH_SIZE_OPT=--batch_size=${BATCH_SIZE}

    echo "Running caffe2 batchsize=${BATCH_SIZE}"

    for i in $(seq 17 17); do
        echo "infer using model $i"
        NSIGHT_PATH="${RESULTS_DIR}/$((i + 1))_${BATCH_SIZE}_${NUM_WARMUP}_${NUM_ITERATIONS}"
        echo "${NSIGHT_PATH}"
        rm -fr ${NSIGHT_PATH}*

        # run onnx models
        if [[ "$i" -eq 0 ]]; then # arcface
            ${NSYS} profile --force-overwrite=true --trace=cuda,cudnn,cublas,nvtx --sample=none --output=${NSIGHT_PATH} --export=sqlite python main.py ${BATCH_SIZE_OPT} --backend=caffe2 --short_output --num_warmup=$NUM_WARMUP --num_iterations=$NUM_ITERATIONS --profile --model_idx=$i --input_dim=112
        elif [[ "$i" -eq 6 ]]; then # duc
            ${NSYS} profile --force-overwrite=true --trace=cuda,cudnn,cublas,nvtx --sample=none --output=${NSIGHT_PATH} --export=sqlite python main.py ${BATCH_SIZE_OPT} --backend=caffe2 --short_output --num_warmup=$NUM_WARMUP --num_iterations=$NUM_ITERATIONS --profile --model_idx=$i --input_dim=800
        elif [[ "$i" -eq 7 ]]; then # emotion_ferplus
            ${NSYS} profile --force-overwrite=true --trace=cuda,cudnn,cublas,nvtx --sample=none --output=${NSIGHT_PATH} --export=sqlite python main.py ${BATCH_SIZE_OPT} --backend=caffe2 --short_output --num_warmup=$NUM_WARMUP --num_iterations=$NUM_ITERATIONS --profile --model_idx=$i --input_dim=64 --input_channels=1
        elif [[ "$i" -eq 10 ]]; then # mnist
            ${NSYS} profile --force-overwrite=true --trace=cuda,cudnn,cublas,nvtx --sample=none --output=${NSIGHT_PATH} --export=sqlite python main.py ${BATCH_SIZE_OPT} --backend=caffe2 --short_output --num_warmup=$NUM_WARMUP --num_iterations=$NUM_ITERATIONS --profile --model_idx=$i --input_dim=28 --input_channels=1
        elif [[ "$i" -eq 24 ]]; then # tiny_yolo
            ${NSYS} profile --force-overwrite=true --trace=cuda,cudnn,cublas,nvtx --sample=none --output=${NSIGHT_PATH} --export=sqlite python main.py ${BATCH_SIZE_OPT} --backend=caffe2 --short_output --num_warmup=$NUM_WARMUP --num_iterations=$NUM_ITERATIONS --profile --model_idx=$i --input_dim=416
        else
            ${NSYS} profile --force-overwrite=true --show-output=true --trace=cuda,cudnn,cublas,nvtx --sample=none --output=${NSIGHT_PATH} --export=sqlite python main.py ${BATCH_SIZE_OPT} --backend=caffe2 --short_output --num_warmup=$NUM_WARMUP --num_iterations=$NUM_ITERATIONS --profile --model_idx=$i
        fi
        gzip ${NSIGHT_PATH}.qdrep
        gzip ${NSIGHT_PATH}.sqlite
    done
done
