"""TensorRT-owned model, reusable CUDA outputs, no framework allocations."""
from pathlib import Path
import numpy as np
import tensorrt as trt
from model_contract import ModelContract


class TensorRTDetector:
    def __init__(self, runtime, engine_path, model_type):
        self.rt=runtime
        self.contract=ModelContract(engine_path,model_type)
        self.model_type=model_type
        self.class_names=self.contract.canonical_labels
        self.logger=trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger,"")
        self.runtime=trt.Runtime(self.logger)
        self.engine=self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError("Could not deserialize engine")
        self.context=self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Could not create execution context")
        self.inputs=[i for i in range(self.engine.num_bindings) if self.engine.binding_is_input(i)]
        self.outputs=[i for i in range(self.engine.num_bindings) if not self.engine.binding_is_input(i)]
        if len(self.inputs)!=1:
            raise ValueError("Exactly one model input is required")
        self.input_shape=tuple(self.engine.get_binding_shape(self.inputs[0]))
        self.output_shapes=[tuple(self.engine.get_binding_shape(i)) for i in self.outputs]
        self.contract.validate_bindings(self.input_shape,self.output_shapes)
        for i in range(self.engine.num_bindings):
            if (self.engine.get_binding_dtype(i)!=trt.float32 or
                    self.engine.get_binding_format(i)!=trt.TensorFormat.LINEAR):
                raise ValueError("Native model bindings must be linear FP32 (internal FP16 is supported)")
        self.output_tensors=[runtime.empty(shape) for shape in self.output_shapes]
        self.bindings=[0]*self.engine.num_bindings
        for i,tensor in zip(self.outputs,self.output_tensors):
            self.bindings[i]=tensor.ptr

    def get_input_resolution(self):
        return self.input_shape[2:]

    def infer(self,tensor):
        if (tensor.shape!=self.input_shape or tensor.dtype!=np.dtype("float32") or
                not tensor.ptr or tensor.nbytes!=int(np.prod(self.input_shape))*4):
            raise ValueError("Input is not the expected contiguous FP32 NCHW tensor")
        self.bindings[self.inputs[0]]=tensor.ptr
        if not self.context.execute_async_v2(self.bindings,self.rt.stream):
            raise RuntimeError("TensorRT execution failed")
        return self.output_tensors

    def close(self):
        self.rt.synchronize()
        for tensor in self.output_tensors:
            tensor.close()
        self.output_tensors.clear()
        self.context=None
        self.engine=None
        self.runtime=None
