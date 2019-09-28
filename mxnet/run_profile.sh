#!/bin/bash

# --delay=10 \
# --capture-range=nvtx \
# -p 'forward@*' \
# -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 \

# /opt/nvidia/nsight-systems-cli/2019.5.1/bin/nsys profile \
#     --trace="cuda,cudnn,cublas" \
#     --show-output=true \
#     --sample=none \
#     --cudabacktrace=true \
#     --force-overwrite=true \
#     -o ~/nvidia_nsight_systems/out.qdstrm python gluon_forward.py --model vgg16 --batch_size=32

/opt/nvidia/nsight-systems-cli/2019.5.1/bin/nsys profile \
    --trace="cuda,cudnn,cublas" \
    --show-output=true \
    --sample=none \
    --cudabacktrace=true \
    --force-overwrite=true \
    -o ~/nvidia_nsight_systems/out.qdstrm python gluon_forward.py --model resnet50_v1 --batch_size=32

# /opt/nvidia/nsight-systems-cli/2019.5.1/bin/nsys profile \
#     --trace="nvtx,cuda,cudnn,cublas" \
#     --show-output=true \
#     --sample=none \
#     --cudabacktrace=true \
#     --force-overwrite=true \
#     -o ~/nvidia_nsight_systems/out.qdstrm python local_forward.py
