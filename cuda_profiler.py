# class cuda_profiler_start():
#     import numba.cuda as cuda
#     cuda.profile_start()


# class cuda_profiler_stop():
#     import numba.cuda as cuda
#     cuda.profile_stop()

import ctypes

_cudart = ctypes.CDLL('libcudart.so')


def cuda_profiler_start():
    # As shown at http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__PROFILER.html,
    # the return value will unconditionally be 0. This check is just in case it changes in 
    # the future.
    ret = _cudart.cudaProfilerStart()
    if ret != 0:
        raise Exception("cudaProfilerStart() returned %d" % ret)

def cuda_profiler_stop():
    ret = _cudart.cudaProfilerStop()
    if ret != 0:
        raise Exception("cudaProfilerStop() returned %d" % ret)
