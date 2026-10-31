"""JPEG host decode; all image transforms and planar packing on GPU."""
import ctypes as C
import cv2
import numpy as np
from .runtime import DeviceTensor


class GPUPreprocessor:
    def __init__(self, runtime, target_size, stream):
        self.runtime, self.stream = runtime, stream
        self.target_h, self.target_w = target_size
        self.source = cv2.cuda_GpuMat()
        self.resized = cv2.cuda_GpuMat()
        self.padded = cv2.cuda_GpuMat()
        self.rgb = cv2.cuda_GpuMat()
        self.floats = cv2.cuda_GpuMat()
        # Generic OutputArray Python bindings return a new header; retaining an
        # empty passed-in header instead silently produces an empty GpuMat.
        self.planar = cv2.cuda.createContinuous(
            self.target_h * 3, self.target_w, cv2.CV_32FC1, cv2.cuda_GpuMat())
        self.planes = [self.planar.rowRange(i*self.target_h, (i+1)*self.target_h) for i in range(3)]
        self.tensor = DeviceTensor(self.planar.cudaPtr(), (1,3,self.target_h,self.target_w),
                                   "float32", self.planar)
        self.host_frame = None
        self.horizontal=runtime.empty((self.target_w,3),"int32")
        self.vertical=runtime.empty((self.target_h,3),"int32")
        self.geometry=None
        self.resize_kernel=runtime.function(runtime.load_kernels(),"resize_compatible")
        self._check_layout()

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

    def _check_layout(self):
        if not self.planar.isContinuous() or self.planar.step != self.target_w * 4:
            raise RuntimeError("OpenCV did not allocate continuous planar storage")
        expected = [self.tensor.ptr + i*self.target_h*self.target_w*4 for i in range(3)]
        if [p.cudaPtr() for p in self.planes] != expected:
            raise RuntimeError("GPU channel views are not TensorRT NCHW planes")

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
        # The previous request has completed before this source buffer is reused.
        self.host_frame = image
        self.source.upload(image, self.stream)
        geometry=(h,w,nh,nw)
        if geometry!=self.geometry:
            self.runtime.upload(self._coefficients(w,nw,self.target_w,True),self.horizontal)
            self.runtime.upload(self._coefficients(h,nh,self.target_h,False),self.vertical)
            self.geometry=geometry
        self.resized.create(nh,nw,cv2.CV_8UC3)
        self.runtime.launch(self.resize_kernel,[
            C.c_void_p(self.source.cudaPtr()),C.c_ulonglong(self.source.step),C.c_int(w),C.c_int(h),
            C.c_void_p(self.resized.cudaPtr()),C.c_ulonglong(self.resized.step),C.c_int(nw),C.c_int(nh),
            C.c_void_p(self.horizontal.ptr),C.c_void_p(self.vertical.ptr)
        ],(nw*nh*3+255)//256)
        self.padded = cv2.cuda.copyMakeBorder(
            self.resized, top, self.target_h-nh-top, left, self.target_w-nw-left,
            cv2.BORDER_CONSTANT, dst=self.padded, value=(114,114,114), stream=self.stream)
        self.rgb = cv2.cuda.cvtColor(self.padded, cv2.COLOR_BGR2RGB,
                                     dst=self.rgb, stream=self.stream)
        self.floats = self.rgb.convertTo(cv2.CV_32F, 1./255., 0., self.stream, self.floats)
        self.planes = list(cv2.cuda.split(self.floats, dst=self.planes, stream=self.stream))
        self._check_layout()
        return self.tensor, dict(orig_h=h,orig_w=w,scale=scale,pad_x=px,pad_y=py,
                                 target_h=self.target_h,target_w=self.target_w)

    def close(self):
        self.runtime.synchronize()
        for name in ("source","resized","padded","rgb","floats"):
            getattr(self,name).release()
        self.planes.clear()
        self.planar.release()
        self.horizontal.close()
        self.vertical.close()
        self.host_frame = None
