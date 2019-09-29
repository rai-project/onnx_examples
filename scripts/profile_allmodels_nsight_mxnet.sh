#!/bin/bash

trap "exit" INT

sudo sh -c 'echo 1 >/proc/sys/kernel/perf_event_paranoid'

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

export MXNET_CUDA_ALLOW_TENSOR_CORE=0
export TF_DISABLE_CUDNN_TENSOR_OP_MATH=0
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0

declare -a batch_sizes=(
    # 1
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
RESULTS_DIR=profile_results/mxnet/nsight/${GPU_NAME}

NSYS=/opt/nvidia/nsight-systems/2019.5.1/bin/nsys
NSIGHT_OPTIONS=--sample=none --capture-range=cudaProfilerAPI

mkdir -p ${RESULTS_DIR}
nvidia-smi -x -q -a >${RESULTS_DIR}/nvidia_smi.xml

for BATCH_SIZE in "${batch_sizes[@]}"; do

    BATCH_SIZE_OPT=--batch_size=${BATCH_SIZE}

    echo "Running MXNET batchsize=${BATCH_SIZE}"

    for i in $(seq 0 29); do
        echo "infer using model $i"
        NSIGHT_PATH="${RESULTS_DIR}/$((i + 1))_${BATCH_SIZE}"
        echo "${NSIGHT_PATH}"
        rm -f ${NSIGHT_PATH}*

        # run mxnet models instead of onnx models for batch size > 1 for some models
        if [[ "$BATCH_SIZE" -ne 1 ]]; then
            if [[ "$i" -eq 1 ]]; then # alexnet
                ${NSYS} profile --trace=cuda,cudnn,cublas,nvtx ${NSIGHT_OPTIONS} --output=${NSIGHT_PATH} --export=sqlite python mxnet/gluon_forward.py ${BATCH_SIZE_OPT} --model_name=alexnet --model_idx=$i --num_warmup=$NUM_WARMUP --num_iterations=$NUM_ITERATIONS --profile=True
                continue
            elif [[ "$i" -eq 2 ]]; then # bvlc_caffenet
                ${NSYS} profile --trace=cuda,cudnn,cublas,nvtx ${NSIGHT_OPTIONS} --output=${NSIGHT_PATH} --export=sqlite python mxnet/local_forward.py ${BATCH_SIZE_OPT} --model_name=bvlc_caffenet --model_idx=$i --num_warmup=$NUM_WARMUP --num_iterations=$NUM_ITERATIONS --profile=True
                continue
            elif [[ "$i" -eq 3 ]]; then # bvlc_googlenet
                ${NSYS} profile --trace=cuda,cudnn,cublas,nvtx ${NSIGHT_OPTIONS} --output=${NSIGHT_PATH} --export=sqlite python mxnet/local_forward.py ${BATCH_SIZE_OPT} --model_name=bvlc_googlenet --model_idx=$i --num_warmup=$NUM_WARMUP --num_iterations=$NUM_ITERATIONS --profile=True
                continue
            elif [[ "$i" -eq 4 ]]; then # bvlc_rcnn_ilsvrc13
                ${NSYS} profile --trace=cuda,cudnn,cublas,nvtx ${NSIGHT_OPTIONS} --output=${NSIGHT_PATH} --export=sqlite python mxnet/local_forward.py ${BATCH_SIZE_OPT} --model_name=bvlc_rcnn_ilsvrc13 --model_idx=$i --num_warmup=$NUM_WARMUP --num_iterations=$NUM_ITERATIONS --profile=True
                continue
            elif [[ "$i" -eq 8 ]]; then # inception_v1
                ${NSYS} profile --trace=cuda,cudnn,cublas,nvtx ${NSIGHT_OPTIONS} --output=${NSIGHT_PATH} --export=sqlite python mxnet/local_forward.py ${BATCH_SIZE_OPT} --model_name=inception_v1 --model_idx=$i --num_warmup=$NUM_WARMUP --num_iterations=$NUM_ITERATIONS --profile=True
                continue
            elif [[ "$i" -eq 29 ]]; then # zfnet512
                ${NSYS} profile --trace=cuda,cudnn,cublas,nvtx ${NSIGHT_OPTIONS} --output=${NSIGHT_PATH} --export=sqlite python mxnet/local_forward.py ${BATCH_SIZE_OPT} --model_name=zfnet512 --model_idx=$i --num_warmup=$NUM_WARMUP --num_iterations=$NUM_ITERATIONS --profile=True
                continue
            elif [[ ("$i" -eq 7) || ("$i" -eq 9) ]]; then
                continue
            fi
        fi

        # run onnx models
        if [[ "$i" -eq 0 ]]; then # arcface
            ${NSYS} profile --trace=cuda,cudnn,cublas,nvtx ${NSIGHT_OPTIONS} --output=${NSIGHT_PATH} --export=sqlite python main.py ${BATCH_SIZE_OPT} --backend=mxnet --short_output --num_warmup=$NUM_WARMUP --num_iterations=$NUM_ITERATIONS --profile --model_idx=$i --input_dim=112
        elif [[ "$i" -eq 7 ]]; then # emotion_ferplus
            ${NSYS} profile --trace=cuda,cudnn,cublas,nvtx ${NSIGHT_OPTIONS} --output=${NSIGHT_PATH} --export=sqlite python main.py ${BATCH_SIZE_OPT} --backend=mxnet --short_output --num_warmup=$NUM_WARMUP --num_iterations=$NUM_ITERATIONS --profile --model_idx=$i --input_dim=64 --input_channels=1
        elif [[ "$i" -eq 10 ]]; then # mnist
            ${NSYS} profile --trace=cuda,cudnn,cublas,nvtx ${NSIGHT_OPTIONS} --output=${NSIGHT_PATH} --export=sqlite python main.py ${BATCH_SIZE_OPT} --backend=mxnet --short_output --num_warmup=$NUM_WARMUP --num_iterations=$NUM_ITERATIONS --profile --model_idx=$i --input_dim=28 --input_channels=1
        elif [[ "$i" -eq 24 ]]; then # tiny_yolo
            ${NSYS} profile --trace=cuda,cudnn,cublas,nvtx ${NSIGHT_OPTIONS} --output=${NSIGHT_PATH} --export=sqlite python main.py ${BATCH_SIZE_OPT} --backend=mxnet --short_output --num_warmup=$NUM_WARMUP --num_iterations=$NUM_ITERATIONS --profile --model_idx=$i --input_dim=416
        else
            ${NSYS} profile --trace=cuda,cudnn,cublas,nvtx ${NSIGHT_OPTIONS} --output=${NSIGHT_PATH} --export=sqlite python main.py ${BATCH_SIZE_OPT} --backend=mxnet --short_output --num_warmup=$NUM_WARMUP --num_iterations=$NUM_ITERATIONS --profile --model_idx=$i
        fi
        # gzip ${NSIGHT_PATH}.qdrep
        # gzip ${NSIGHT_PATH}.sqlite
    done
done
