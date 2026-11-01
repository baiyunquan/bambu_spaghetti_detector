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
        ptr = C.c_uint64()
        shape = tuple(int(n) for n in shape)
        if not shape or any(n <= 0 for n in shape):
            raise ValueError("Invalid allocation dimensions")
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        runtime.driver_check(runtime.driver.cuMemAlloc_v2(C.byref(ptr), size))
        super().__init__(ptr.value, shape, dtype, runtime)
        runtime.buffers.add(self)

    def close(self):
        if self.ptr:
            self.owner.driver_check(self.owner.driver.cuMemFree_v2(self.ptr))
            self.ptr = 0
            self.owner.buffers.discard(self)


class Runtime:
    def __init__(self):
        self.driver = C.CDLL("libcuda.so.1")
        self.buffers = set()
        self.modules = []
        self.kernel_modules = {}
        self.stream = 0
        self.context = C.c_void_p()
        self.previous_context = C.c_void_p()
        self.closed = False
        for name, args in [
            ("cuInit", [C.c_uint]),
            ("cuCtxGetCurrent", [C.POINTER(C.c_void_p)]),
            ("cuCtxSetCurrent", [C.c_void_p]),
            ("cuDevicePrimaryCtxRetain", [C.POINTER(C.c_void_p), C.c_int]),
            ("cuDevicePrimaryCtxRelease", [C.c_int]),
            ("cuStreamCreate", [C.POINTER(C.c_void_p), C.c_uint]),
            ("cuStreamDestroy_v2", [C.c_void_p]),
            ("cuStreamSynchronize", [C.c_void_p]),
            ("cuMemAlloc_v2", [C.POINTER(C.c_uint64), C.c_size_t]),
            ("cuMemFree_v2", [C.c_uint64]),
            ("cuMemcpyHtoD_v2", [C.c_uint64, C.c_void_p, C.c_size_t]),
            ("cuMemcpyDtoH_v2", [C.c_void_p, C.c_uint64, C.c_size_t]),
            ("cuMemsetD8Async", [C.c_uint64, C.c_ubyte, C.c_size_t, C.c_void_p]),
            ("cuGetErrorString", [C.c_int, C.POINTER(C.c_char_p)]),
            ("cuModuleLoadData", [C.POINTER(C.c_void_p), C.c_void_p]),
            ("cuModuleGetFunction", [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p]),
            ("cuModuleUnload", [C.c_void_p]),
            ("cuLaunchKernel", [C.c_void_p] + [C.c_uint] * 7 +
             [C.c_void_p, C.POINTER(C.c_void_p), C.c_void_p]),
        ]:
            fn = getattr(self.driver, name)
            fn.argtypes, fn.restype = args, C.c_int
        self.driver_check(self.driver.cuInit(0))
        self.driver_check(self.driver.cuCtxGetCurrent(C.byref(self.previous_context)))
        self.driver_check(self.driver.cuDevicePrimaryCtxRetain(C.byref(self.context), 0))
        try:
            # Retain the same device primary context TensorRT/OpenCV use, never
            # create a secondary context. Avoid initializing OpenCV CUDA: on
            # CUDA 10.2 its eager module registration consumes about 500 MiB.
            self.driver_check(self.driver.cuCtxSetCurrent(self.context))
            stream = C.c_void_p()
            self.driver_check(self.driver.cuStreamCreate(C.byref(stream), 0))
            self.stream = int(stream.value)
        except Exception:
            self.driver.cuCtxSetCurrent(self.previous_context)
            self.driver.cuDevicePrimaryCtxRelease(0)
            raise

    def driver_check(self, status):
        if status:
            message = C.c_char_p()
            self.driver.cuGetErrorString(status, C.byref(message))
            raise RuntimeError("CUDA driver {}: {}".format(
                status, message.value.decode() if message.value else "?"))

    def empty(self, shape, dtype="float32"):
        return DeviceBuffer(self, shape, dtype)

    def synchronize(self):
        if not self.closed:
            self.driver_check(self.driver.cuStreamSynchronize(C.c_void_p(self.stream)))

    def zero(self, tensor):
        self.driver_check(self.driver.cuMemsetD8Async(
            tensor.ptr, 0, tensor.nbytes, C.c_void_p(self.stream)))

    def upload(self, array, tensor=None):
        array = np.ascontiguousarray(array)
        if tensor is None:
            tensor = self.empty(array.shape, array.dtype)
        if array.nbytes != tensor.nbytes or array.dtype != tensor.dtype:
            raise ValueError("Host/device buffer mismatch")
        # Driver synchronous copies accept pageable JPEG/NumPy storage. Complete
        # the single stream first; no implicit default-stream ordering assumed.
        self.synchronize()
        self.driver_check(self.driver.cuMemcpyHtoD_v2(
            tensor.ptr, C.c_void_p(array.ctypes.data), array.nbytes))
        return tensor

    def download(self, tensor, count=None):
        shape = tensor.shape if count is None else (int(count),)
        array = np.empty(shape, dtype=tensor.dtype)
        if array.nbytes > tensor.nbytes:
            raise ValueError("Device download exceeds allocation")
        self.synchronize()
        self.driver_check(self.driver.cuMemcpyDtoH_v2(
            C.c_void_p(array.ctypes.data), tensor.ptr, array.nbytes))
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
        if self.closed:
            return
        error = None
        actions = [self.synchronize]
        actions += [buffer.close for buffer in list(self.buffers)]
        actions += [lambda module=module: self.driver_check(self.driver.cuModuleUnload(module))
                    for module in self.modules]
        actions += [lambda: self.driver_check(self.driver.cuStreamDestroy_v2(C.c_void_p(self.stream))),
                    lambda: self.driver_check(self.driver.cuCtxSetCurrent(self.previous_context)),
                    lambda: self.driver_check(self.driver.cuDevicePrimaryCtxRelease(0))]
        for action in actions:
            try:
                action()
            except Exception as exc:
                if error is None:
                    error = exc
        self.modules.clear()
        self.kernel_modules.clear()
        self.closed = True
        self.stream = 0
        if error is not None:
            raise error
