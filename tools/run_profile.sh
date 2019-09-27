#!/bin/bash

# --delay=10 \
# --capture-range=nvtx \
# -p 'forward@*' \
# -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 \

/opt/nvidia/nsight-systems-cli/2019.5.1/bin/nsys profile \
    --trace="cuda,cudnn,cublas" \
    --stats=true \
    --show-output=true \
    --sample=none \
    --force-overwrite=true \
    -o ~/nvidia_nsight_systems/ python forward.py --model vgg16 --batch_size=32
