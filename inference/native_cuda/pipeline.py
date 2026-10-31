"""One model, one shared primary CUDA context and stream, bounded buffers."""
import sys
import cv2
import numpy as np
from .runtime import Runtime
from .preprocess import GPUPreprocessor
from .postprocess import GPUPostProcessor
from .detector import TensorRTDetector


class NativePipeline:
    def __init__(self,engine_path,model_type,conf_thresh=.35,iou_thresh=.45):
        self.rt=Runtime()
        self.stream=cv2.cuda_Stream()
        self.rt.stream=int(self.stream.cudaPtr())
        self.detector=self.pre=self.post=None
        try:
            self.detector=TensorRTDetector(self.rt,engine_path,model_type)
            self.pre=GPUPreprocessor(self.rt,self.detector.get_input_resolution(),self.stream)
            shape=self.detector.output_shapes[0]
            n=shape[2] if model_type=="yolov8" else shape[1]
            self.post=GPUPostProcessor(self.rt,n,self.detector.class_names,model_type,conf_thresh,iou_thresh)
        except Exception:
            self.close()
            raise
        self._assert_no_torch()

    @staticmethod
    def _assert_no_torch():
        if any(k=="torch" or k.startswith("torch.") or k=="torchvision" or
               k.startswith("torchvision.") for k in sys.modules):
            raise RuntimeError("Native worker imported torch/torchvision")

    def detect(self,payload):
        try:
            tensor,meta=self.pre.decode_and_preprocess(payload)
            outputs=self.detector.infer(tensor)
            return self.post.process(outputs,meta)
        finally:
            # Both success and failures complete uses of the frame's host/device
            # buffers before another request can replace them.
            self.rt.synchronize()

    def warmup(self):
        for h,w in ((720,1280),(1080,1536)):
            image=np.random.RandomState(0).randint(0,256,(h,w,3),dtype=np.uint8)
            ok,jpg=cv2.imencode(".jpg",image)
            if not ok: raise RuntimeError("Could not create warmup JPEG")
            for _ in range(3): self.detect(jpg.tobytes())
        # Reuse model outputs to force positive NMS without another model/context.
        raw=[]
        for shape in self.detector.output_shapes:
            raw.append(np.zeros(shape,dtype=np.float32))
        kind=self.detector.model_type
        if kind=="obico":
            raw[0][0,:3,0]=[[.1,.1,.5,.5],[.12,.12,.48,.48],[.7,.7,.9,.9]]
            raw[1][0,:3,0]=[1.,.9,.8]
        elif kind=="yolov8":
            raw[0][0,:4,:3]=[[100,102,300],[100,102,300],[50,50,50],[50,50,50]]
            raw[0][0,4,:3]=[1.,.9,.8]
        else:
            raw[0][0,:3,:5]=[[100,100,50,50,1],[102,102,50,50,1],[300,300,50,50,1]]
            raw[0][0,:3,5+self.detector.class_names.index("spaghetti")]=[1.,.9,.8]
        for array,tensor in zip(raw,self.detector.output_tensors):
            self.rt.upload(array,tensor)
        h,w=self.detector.get_input_resolution()
        self.post.process(self.detector.output_tensors,
                          dict(target_h=h,target_w=w,orig_h=h,orig_w=w,scale=1.,pad_x=0.,pad_y=0.))
        self.rt.synchronize()
        self._assert_no_torch()

    def close(self):
        try:
            if self.post is not None: self.post.close()
            if self.pre is not None: self.pre.close()
            if self.detector is not None: self.detector.close()
        finally:
            self.rt.close()
            self.post=self.pre=self.detector=None
