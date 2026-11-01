"""JPEG host decode; all image transforms and planar packing on GPU."""
import ctypes as C
import cv2
import numpy as np


class GPUPreprocessor:
    def __init__(self, runtime, target_size, stream=None):
        self.runtime = runtime
        self.target_h, self.target_w = target_size
        self.source = None
        self.tensor = runtime.empty((1,3,self.target_h,self.target_w))
        self.resized = runtime.empty((self.target_h*self.target_w*3,), "uint8")
        self.horizontal = runtime.empty((self.target_w,3), "int32")
        self.vertical = runtime.empty((self.target_h,3), "int32")
        self.geometry = None
        module = runtime.load_kernels()
        self.resize_kernel = runtime.function(module,"resize_compatible")
        self.pack_kernel = runtime.function(module,"letterbox_rgb_nchw")

    @staticmethod
    def _coefficients(source,length,capacity,horizontal):
        positions=((np.arange(length,dtype=np.float64)+.5)*
                   (1./(float(length)/source))-.5).astype(np.float32)
        indices=np.floor(positions).astype(np.int32)
        fraction=positions-indices.astype(np.float32)
        if horizontal:
            edge=(indices<0)|(indices>=source-1)
            indices=np.clip(indices,0,source-1)
            fraction[edge]=0
        table=np.zeros((capacity,3),dtype=np.int32)
        table[:length,0]=indices
        table[:length,1]=np.rint((np.float32(1.)-fraction)*np.float32(2048)).astype(np.int32)
        table[:length,2]=np.rint(fraction*np.float32(2048)).astype(np.int32)
        return table

    def decode_and_preprocess(self, payload):
        image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode JPEG bytes")
        h,w = image.shape[:2]
        scale = min(self.target_w / w, self.target_h / h)
        nw,nh = int(round(w*scale)),int(round(h*scale))
        if nw<1 or nh<1:
            raise ValueError("Unsupported image aspect ratio")
        px,py = (self.target_w-nw)/2.,(self.target_h-nh)/2.
        left,top = int(round(px-.1)),int(round(py-.1))
        # One source allocation for the current size only. Complete outstanding
        # uses before replacing it; retain no historical size/frame cache.
        if self.source is None or self.source.shape != image.shape:
            self.runtime.synchronize()
            if self.source is not None:
                self.source.close()
                self.source = None
            self.source = self.runtime.empty(image.shape,"uint8")
        self.runtime.upload(image,self.source)
        geometry=(h,w,nh,nw)
        if geometry!=self.geometry:
            self.runtime.upload(self._coefficients(w,nw,self.target_w,True),self.horizontal)
            self.runtime.upload(self._coefficients(h,nh,self.target_h,False),self.vertical)
            self.geometry=geometry
        self.runtime.launch(self.resize_kernel,[
            C.c_void_p(self.source.ptr),C.c_ulonglong(w*3),C.c_int(w),C.c_int(h),
            C.c_void_p(self.resized.ptr),C.c_ulonglong(nw*3),C.c_int(nw),C.c_int(nh),
            C.c_void_p(self.horizontal.ptr),C.c_void_p(self.vertical.ptr)
        ],(nw*nh*3+255)//256)
        self.runtime.launch(self.pack_kernel,[
            C.c_void_p(self.resized.ptr),C.c_void_p(self.tensor.ptr),C.c_int(nw),C.c_int(nh),
            C.c_int(self.target_w),C.c_int(self.target_h),C.c_int(left),C.c_int(top)
        ],(self.target_w*self.target_h*3+255)//256)
        return self.tensor, dict(orig_h=h,orig_w=w,scale=scale,pad_x=px,pad_y=py,
                                 target_h=self.target_h,target_w=self.target_w)

    def close(self):
        self.runtime.synchronize()
        for name in ("source","resized","tensor","horizontal","vertical"):
            buffer = getattr(self,name)
            if buffer is not None:
                buffer.close()
