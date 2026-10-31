"""Minimal, explicitly typed CUDA 10.2 bindings; share the primary context."""
import ctypes as C
import hashlib
import json
from pathlib import Path
import numpy as np


class DeviceTensor:
    def __init__(self, ptr, shape, dtype, owner):
        self.ptr = int(ptr)
        self.shape = tuple(int(n) for n in shape)
        self.dtype = np.dtype(dtype)
        self.owner = owner
        if not self.ptr or not self.shape or any(n <= 0 for n in self.shape):
            raise ValueError("Invalid device tensor pointer or dimensions")
        self.nbytes = int(np.prod(self.shape)) * self.dtype.itemsize


class DeviceBuffer(DeviceTensor):
    def __init__(self, runtime, shape, dtype):
        ptr = C.c_void_p()
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        runtime.check(runtime.lib.cudaMalloc(C.byref(ptr), size))
        super().__init__(ptr.value, shape, dtype, runtime)
        runtime.buffers.add(self)

    def close(self):
        if self.ptr:
            self.owner.check(self.owner.lib.cudaFree(C.c_void_p(self.ptr)))
            self.ptr = 0
            self.owner.buffers.discard(self)


class Runtime:
    def __init__(self):
        self.lib = C.CDLL("/usr/local/cuda/lib64/libcudart.so")
        self.driver = C.CDLL("libcuda.so.1")
        self.buffers = set()
        self.modules = []
        self.kernel_modules = {}
        self.stream = 0
        for name, args, restype in [
            ("cudaSetDevice", [C.c_int], C.c_int),
            ("cudaMalloc", [C.POINTER(C.c_void_p), C.c_size_t], C.c_int),
            ("cudaFree", [C.c_void_p], C.c_int),
            ("cudaMemcpyAsync", [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int, C.c_void_p], C.c_int),
            ("cudaMemsetAsync", [C.c_void_p, C.c_int, C.c_size_t, C.c_void_p], C.c_int),
            ("cudaStreamSynchronize", [C.c_void_p], C.c_int),
            ("cudaGetErrorString", [C.c_int], C.c_char_p),
        ]:
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, restype
        for name, args in [
            ("cuInit", [C.c_uint]),
            ("cuModuleLoadData", [C.POINTER(C.c_void_p), C.c_void_p]),
            ("cuModuleGetFunction", [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p]),
            ("cuModuleUnload", [C.c_void_p]),
            ("cuLaunchKernel", [C.c_void_p] + [C.c_uint] * 7 +
             [C.c_void_p, C.POINTER(C.c_void_p), C.c_void_p]),
        ]:
            fn = getattr(self.driver, name)
            fn.argtypes, fn.restype = args, C.c_int
        self.driver_check(self.driver.cuInit(0))
        self.check(self.lib.cudaSetDevice(0))
        # CUDA Runtime and OpenCV use this same device primary context. Never
        # create a second context (e.g. pycuda.autoinit).
        self.check(self.lib.cudaFree(C.c_void_p()))

    def check(self, status):
        if status:
            message = self.lib.cudaGetErrorString(status)
            raise RuntimeError("CUDA {}: {}".format(status, message.decode() if message else "?"))

    @staticmethod
    def driver_check(status):
        if status:
            raise RuntimeError("CUDA driver error {}".format(status))

    def empty(self, shape, dtype="float32"):
        return DeviceBuffer(self, shape, dtype)

    def synchronize(self):
        self.check(self.lib.cudaStreamSynchronize(C.c_void_p(self.stream)))

    def zero(self, tensor):
        self.check(self.lib.cudaMemsetAsync(C.c_void_p(tensor.ptr), 0,
                                          tensor.nbytes, C.c_void_p(self.stream)))

    def upload(self, array, tensor=None):
        array = np.ascontiguousarray(array)
        if tensor is None:
            tensor = self.empty(array.shape, array.dtype)
        if array.nbytes != tensor.nbytes or array.dtype != tensor.dtype:
            raise ValueError("Host/device buffer mismatch")
        self.check(self.lib.cudaMemcpyAsync(C.c_void_p(tensor.ptr),
                                           C.c_void_p(array.ctypes.data), array.nbytes,
                                           1, C.c_void_p(self.stream)))
        # Uploads here are metadata/test fixtures, not the video upload path.
        self.synchronize()
        return tensor

    def download(self, tensor, count=None):
        shape = tensor.shape if count is None else (int(count),)
        array = np.empty(shape, dtype=tensor.dtype)
        if array.nbytes > tensor.nbytes:
            raise ValueError("Device download exceeds allocation")
        self.check(self.lib.cudaMemcpyAsync(C.c_void_p(array.ctypes.data),
                                           C.c_void_p(tensor.ptr), array.nbytes,
                                           2, C.c_void_p(self.stream)))
        self.synchronize()
        return array

    def load_kernels(self, directory=None):
        folder = Path(directory) if directory else Path(__file__).parent / "build"
        key=str(folder.resolve())
        if key in self.kernel_modules:
            return self.kernel_modules[key]
        source = Path(__file__).with_name("kernels.cu").read_bytes()
        manifest = json.loads((folder / "kernels.json").read_text())
        ptx = (folder / "kernels.ptx").read_bytes()
        if (manifest["source_sha256"] != hashlib.sha256(source).hexdigest() or
                manifest["ptx_sha256"] != hashlib.sha256(ptx).hexdigest() or
                manifest["architecture"] != "compute_53"):
            raise RuntimeError("Stale CUDA kernels; run native_cuda.build_kernels on Jetson")
        module = C.c_void_p()
        data = C.create_string_buffer(ptx)
        self.driver_check(self.driver.cuModuleLoadData(C.byref(module), C.cast(data, C.c_void_p)))
        self.modules.append(module)
        self.kernel_modules[key]=module
        return module

    def function(self, module, name):
        ptr = C.c_void_p()
        self.driver_check(self.driver.cuModuleGetFunction(C.byref(ptr), module, name.encode()))
        return ptr

    def launch(self, function, arguments, grid, block=256, shared=0):
        grid = (grid, 1, 1) if isinstance(grid, int) else grid
        block = (block, 1, 1) if isinstance(block, int) else block
        argv = (C.c_void_p * len(arguments))(
            *[C.cast(C.byref(arg), C.c_void_p) for arg in arguments])
        self.driver_check(self.driver.cuLaunchKernel(
            function, *grid, *block, shared, C.c_void_p(self.stream), argv, None))

    def close(self):
        try:
            self.synchronize()
        finally:
            for buffer in list(self.buffers):
                buffer.close()
            for module in self.modules:
                self.driver_check(self.driver.cuModuleUnload(module))
            self.modules.clear()
            self.kernel_modules.clear()
