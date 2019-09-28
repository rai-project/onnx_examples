#!/bin/bash

mkdir -p nvidia_nsight_systems

# --delay=10 \
# --capture-range=nvtx \
# -p 'forward@*' \
# -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 \
    # --capture-range=nvtx \

py=`pyenv which python`

/opt/nvidia/nsight-systems/2019.5.1/target-linux-x64/nsys profile \
    --wait=all \
    --stats=true \
    --trace=cuda,cublas,cudnn \
    --backtrace=none \
    --export=sqlite \
    python forward.py --model vgg16 --batch_size=32
