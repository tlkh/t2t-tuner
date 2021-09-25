import ctypes
from re import X
import torch
from transformers import TrainerCallback


def round_up(x, to=8):
    x, to = int(x), int(to)
    return int((x + to - 1) & (-1 * to))


def increase_l2_fetch_granularity():
    # see:
    # https://www.tensorflow.org/guide/profiler#max_out_the_l2_cache
    # "A value between 0 and 128 that indicates the maximum fetch granularity of L2 (in Bytes)."
    # https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html
    # maximum fetch granularity of L2: 128 bytes
    _libcudart = ctypes.CDLL('libcudart.so')
    # set device limit on the current device
    # cudaLimitMaxL2FetchGranularity = 0x05
    pValue = ctypes.cast((ctypes.c_int * 1)(), ctypes.POINTER(ctypes.c_int))
    _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))


class L2FetchCallback(TrainerCallback):
    "Set cudaLimitMaxL2FetchGranularity at beginning of training"

    def on_train_begin(self, args, state, control, **kwargs):
        increase_l2_fetch_granularity()
