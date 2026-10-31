"""
Unified TensorRT 8.2 Inference Engine for Jetson Nano (Python 3.6 / CUDA 10.2)
==============================================================================
Loads TensorRT FP16 engine, maps memory buffers directly to PyTorch CUDA tensors
(zero Host-to-Device copy), and executes asynchronous forward inference.
"""

import os
from typing import List, Dict, Any, Tuple
import tensorrt as trt
import torch


class TensorRTDetector:
    def __init__(self, engine_path: str, model_type: str = "obico", class_names: List[str] = None):
        """
        Initialize TensorRT Engine.

        Args:
            engine_path: Path to .engine plan file.
            model_type: "obico", "yolov8", or "yolov5".
            class_names: List of class label names.
        """
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TensorRT engine not found at: {engine_path}")

        self.engine_path = engine_path
        if class_names:
            self.class_names = class_names
        elif self.model_type == "obico":
            self.class_names = ["spaghetti"]
        elif self.model_type == "yolov5":
            self.class_names = ["error", "extrusor", "part", "spaghetti"]
        else:
            self.class_names = ["spaghetti", "warping"]

        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, "")

        # Load engine
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context")

        # Inspect bindings
        self.input_bindings = []
        self.output_bindings = []
        self.input_shape = None

        self._inspect_bindings()

    def _inspect_bindings(self):
        """Parse input and output bindings from engine."""
        num_bindings = self.engine.num_bindings
        for idx in range(num_bindings):
            name = self.engine.get_binding_name(idx)
            is_input = self.engine.binding_is_input(idx)
            shape = tuple(self.engine.get_binding_shape(idx))
            dtype = self.engine.get_binding_dtype(idx)
            torch_dtype = self._trt_dtype_to_torch(dtype)

            binding_info = {
                "index": idx,
                "name": name,
                "shape": shape,
                "dtype": torch_dtype
            }

            if is_input:
                self.input_bindings.append(binding_info)
                if self.input_shape is None:
                    self.input_shape = shape
            else:
                self.output_bindings.append(binding_info)

    @staticmethod
    def _trt_dtype_to_torch(trt_dtype):
        if trt_dtype == trt.float32:
            return torch.float32
        elif trt_dtype == trt.float16:
            return torch.float16
        elif trt_dtype == trt.int32:
            return torch.int32
        elif trt_dtype == trt.int8:
            return torch.int8
        elif trt_dtype == trt.bool:
            return torch.bool
        return torch.float32

    def get_input_resolution(self) -> Tuple[int, int]:
        """Returns (height, width) expected by engine."""
        # Typically shape is (1, 3, H, W) or (B, C, H, W)
        if len(self.input_shape) == 4:
            return (self.input_shape[2], self.input_shape[3])
        return (416, 416)

    def infer(self, input_tensor: torch.Tensor) -> List[torch.Tensor]:
        """
        Executes inference on GPU using PyTorch CUDA tensors directly.

        Args:
            input_tensor: PyTorch CUDA tensor of shape (1, 3, H, W) matching input_shape.

        Returns:
            List of output PyTorch CUDA tensors.
        """
        if tuple(input_tensor.shape) != self.input_shape:
            raise ValueError("Input shape {} does not match engine {}".format(
                tuple(input_tensor.shape), self.input_shape))
        expected_dtype = self.input_bindings[0]["dtype"]
        if input_tensor.dtype != expected_dtype:
            raise ValueError("Input dtype {} does not match engine {}".format(
                input_tensor.dtype, expected_dtype))

        if not input_tensor.is_cuda:
            input_tensor = input_tensor.cuda()
        # Callers other than GPUPreprocessor may supply a strided view. TRT's
        # linear input binding cannot interpret its strides.
        input_tensor = input_tensor.contiguous()

        # Allocate output GPU tensors
        output_tensors = []
        for out_b in self.output_bindings:
            out_tensor = torch.empty(out_b["shape"], dtype=out_b["dtype"], device=input_tensor.device)
            output_tensors.append(out_tensor)

        # Build bindings pointer list
        bindings = [None] * self.engine.num_bindings
        for inp_b in self.input_bindings:
            bindings[inp_b["index"]] = int(input_tensor.data_ptr())
        for idx, out_b in enumerate(self.output_bindings):
            bindings[out_b["index"]] = int(output_tensors[idx].data_ptr())

        # Execute asynchronously on current CUDA stream
        stream = torch.cuda.current_stream()
        success = self.context.execute_async_v2(bindings=bindings, stream_handle=stream.cuda_stream)
        if not success:
            raise RuntimeError("TensorRT execute_async_v2 failed")

        stream.synchronize()
        return output_tensors
