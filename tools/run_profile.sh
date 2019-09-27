#!/bin/bash

mkdir -p nvidia_nsight_systems

# --delay=10 \
# --capture-range=nvtx \
# -p 'forward@*' \
# -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 \
    # --capture-range=nvtx \

py=`pyenv which python`

/opt/nvidia/nsight-systems/2019.4.2/target-linux-x64/nsys profile \
    --trace="cuda,cudnn,cublas" \
    --backtrace=none \
    --export=sqlite \
    --wait=all \
    ${py} forward.py --model vgg16 --batch_size=32
