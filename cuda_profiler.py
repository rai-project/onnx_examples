class cuda_profiler_start():
    import numba.cuda as cuda
    cuda.profile_start()

class cuda_profiler_stop():
    import numba.cuda as cuda
    cuda.profile_stop()
