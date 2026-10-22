"""Use the pinned MAA FastDeploy preprocessor without changing inference settings."""
import ctypes
from collections import defaultdict
from pathlib import Path
import numpy as np
from file_utils import sha
from workspace import RT
from workspace import RELEASE

class ReleasedInput:
    def __init__(self):
        """加载固定发布版的图像预处理实现与身份。"""
        self.path=RT/'build/libmaa_rec_input.so'
        self.lib=ctypes.CDLL(str(self.path))
        self.lib.maa_rec_input.argtypes=[ctypes.c_char_p,ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int)]
        self.lib.maa_rec_input.restype=ctypes.c_int
        self.lib.maa_input_free.argtypes=[ctypes.POINTER(ctypes.c_float)]
        self.lib.maa_input_error.restype=ctypes.c_char_p
    def __call__(self,path):
        """按 MAA 发布接口把图像转换为模型输入张量。"""
        data=ctypes.POINTER(ctypes.c_float)();height=ctypes.c_int();width=ctypes.c_int()
        rc=self.lib.maa_rec_input(str(Path(path).resolve()).encode(),ctypes.byref(data),ctypes.byref(height),ctypes.byref(width))
        if rc:raise ValueError(self.lib.maa_input_error().decode())
        try:
            # The native library owns this allocation. Copy before freeing it so
            # batches never retain a NumPy view into released native memory.
            return np.ctypeslib.as_array(data,shape=(3*height.value*width.value,)).reshape(3,height.value,width.value).copy()
        finally:self.lib.maa_input_free(data)
    def identity(self):
        """返回预处理源码和运行库身份供冻结计划记录。"""
        files=[self.path,RELEASE/'libfastdeploy_ppocr.so',RELEASE/'libopencv_world4.so.412']
        return {str(p):sha(p) for p in files}

def width_groups(indices,widths):
    """按实际宽度分批，保留所有样本且不把短图补到长图宽度。"""
    groups=defaultdict(list)
    for i in indices:groups[int(widths[int(i)])].append(int(i))
    return list(groups.values())
