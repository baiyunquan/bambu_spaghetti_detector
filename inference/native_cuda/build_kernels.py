"""Build PTX with bundled NVRTC; no CUDA context, torch, or host GCC needed."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path


def build(output=None):
    source_path = Path(__file__).with_name("kernels.cu")
    source = source_path.read_bytes()
    folder = Path(output) if output else source_path.parent / "build"
    lib = C.CDLL("/usr/local/cuda/lib64/libnvrtc.so")
    signatures = {
        "nvrtcCreateProgram": [C.POINTER(C.c_void_p), C.c_char_p, C.c_char_p, C.c_int, C.c_void_p, C.c_void_p],
        "nvrtcCompileProgram": [C.c_void_p, C.c_int, C.POINTER(C.c_char_p)],
        "nvrtcGetProgramLogSize": [C.c_void_p, C.POINTER(C.c_size_t)],
        "nvrtcGetProgramLog": [C.c_void_p, C.c_void_p],
        "nvrtcGetPTXSize": [C.c_void_p, C.POINTER(C.c_size_t)],
        "nvrtcGetPTX": [C.c_void_p, C.c_void_p],
        "nvrtcDestroyProgram": [C.POINTER(C.c_void_p)],
    }
    for name, args in signatures.items():
        getattr(lib, name).argtypes = args
        getattr(lib, name).restype = C.c_int
    program = C.c_void_p()
    status = lib.nvrtcCreateProgram(C.byref(program), source, b"kernels.cu", 0, None, None)
    if status:
        raise RuntimeError("nvrtcCreateProgram failed: {}".format(status))
    try:
        options = (C.c_char_p * 3)(b"--gpu-architecture=compute_53", b"--std=c++11", b"--fmad=false")
        status = lib.nvrtcCompileProgram(program, 3, options)
        if status:
            size = C.c_size_t()
            lib.nvrtcGetProgramLogSize(program, C.byref(size))
            log = C.create_string_buffer(size.value)
            lib.nvrtcGetProgramLog(program, log)
            raise RuntimeError(log.value.decode())
        size = C.c_size_t()
        if lib.nvrtcGetPTXSize(program, C.byref(size)):
            raise RuntimeError("nvrtcGetPTXSize failed")
        ptx = C.create_string_buffer(size.value)
        if lib.nvrtcGetPTX(program, ptx):
            raise RuntimeError("nvrtcGetPTX failed")
        folder.mkdir(parents=True, exist_ok=True)
        data = ptx.raw
        (folder / "kernels.ptx").write_bytes(data)
        (folder / "kernels.json").write_text(json.dumps({
            "architecture": "compute_53",
            "source_sha256": hashlib.sha256(source).hexdigest(),
            "ptx_sha256": hashlib.sha256(data).hexdigest(),
        }, indent=2) + "\n")
        print("Built {} ({} bytes)".format(folder / "kernels.ptx", len(data)))
    finally:
        lib.nvrtcDestroyProgram(C.byref(program))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    build(parser.parse_args().output)
