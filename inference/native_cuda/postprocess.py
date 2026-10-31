"""GPU decode, stable sorting and complete per-class NMS without torch."""
import ctypes as C
import numpy as np

from .runtime import DeviceTensor

CANDIDATE = np.dtype([("box","<f4",4),("score","<f4"),("class_id","<i4"),("anchor","<i4")])


class GPUPostProcessor:
    def __init__(self, runtime, capacity, class_names, model_type, conf_thresh=.35, iou_thresh=.45):
        if not 0 <= conf_thresh <= 1 or not 0 <= iou_thresh <= 1:
            raise ValueError("Confidence and IoU thresholds must be in [0,1]")
        if not 0 < capacity <= 32768:
            raise ValueError("Unsupported candidate capacity (max 32768, no truncation)")
        self.rt,self.capacity,self.class_names = runtime,capacity,list(class_names)
        self.model_type,self.conf_thresh,self.iou_thresh=model_type,conf_thresh,iou_thresh
        self.padded=1 << (capacity-1).bit_length()
        self.stride=(capacity+63)//64
        self.candidates=runtime.empty((self.padded*28,), "uint8")
        self.mask=runtime.empty((capacity*self.stride,), "uint64")
        self.keep=runtime.empty((capacity,), "int32")
        self.result=runtime.empty((capacity*28,), "uint8")
        self.counters=runtime.empty((2,), "int32")
        self.count=runtime.empty((1,), "int32")
        module=runtime.load_kernels()
        self.functions={name:runtime.function(module,name) for name in
                        ("decode","pad","bitonic","overlap_mask","select_indices","gather")}

    def process(self, outputs, meta):
        if self.model_type=="obico":
            n=outputs[0].shape[1]
            score_ptr=outputs[1].ptr
        elif self.model_type=="yolov8":
            n=outputs[0].shape[2]
            score_ptr=0
        else:
            n=outputs[0].shape[1]
            score_ptr=0
        if n!=self.capacity or any(out.dtype!=np.dtype("float32") for out in outputs):
            raise ValueError("Unexpected model output layout/type")
        rt=self.rt
        ptr=lambda tensor:C.c_void_p(tensor.ptr)
        rt.zero(self.counters)
        rt.launch(self.functions["decode"],[
            ptr(outputs[0]),C.c_void_p(score_ptr),ptr(self.candidates),ptr(self.counters),
            C.c_int(n),C.c_int(len(self.class_names)),
            C.c_int({"obico":0,"yolov8":1,"yolov5":2}[self.model_type]),
            C.c_float(self.conf_thresh),C.c_int(meta["target_w"]),C.c_int(meta["target_h"])
        ],(n+255)//256)
        counters=rt.download(self.counters)
        if counters[1]:
            raise ValueError("Non-finite model output")
        k=int(counters[0])
        if k==0:
            return []
        padded=1 << (k-1).bit_length()
        rt.launch(self.functions["pad"],[ptr(self.candidates),C.c_int(k),C.c_int(padded)],
                  (padded+255)//256)
        stage=2
        while stage<=padded:
            step=stage//2
            while step:
                rt.launch(self.functions["bitonic"],[
                    ptr(self.candidates),C.c_int(padded),C.c_int(step),C.c_int(stage)
                ],(padded+255)//256)
                step//=2
            stage*=2
        words=(k+63)//64
        rt.launch(self.functions["overlap_mask"],[
            ptr(self.candidates),ptr(self.mask),C.c_int(k),C.c_int(self.stride),
            C.c_float(self.iou_thresh)],(words,words,1),64)
        rt.launch(self.functions["select_indices"],[
            ptr(self.mask),ptr(self.keep),ptr(self.count),C.c_int(k),C.c_int(self.stride)],1)
        count=int(rt.download(self.count)[0])
        rt.launch(self.functions["gather"],[
            ptr(self.candidates),ptr(self.keep),ptr(self.result),C.c_int(count)
        ],(count+255)//256)
        raw=rt.download(self.result,count=count*28).view(CANDIDATE)
        boxes=raw["box"].copy()
        # Match the original float32 coordinate arithmetic and Python rounding.
        boxes[:,[0,2]]=(boxes[:,[0,2]]-np.float32(meta["pad_x"]))/np.float32(meta["scale"])
        boxes[:,[1,3]]=(boxes[:,[1,3]]-np.float32(meta["pad_y"]))/np.float32(meta["scale"])
        boxes[:,[0,2]]=np.clip(boxes[:,[0,2]],0,meta["orig_w"])
        boxes[:,[1,3]]=np.clip(boxes[:,[1,3]],0,meta["orig_h"])
        boxes=np.rint(boxes).astype(np.int32)
        return [dict(label=self.class_names[int(row["class_id"])],
                     class_id=int(row["class_id"]),confidence=round(float(row["score"]),4),
                     bbox=box.tolist()) for row,box in zip(raw,boxes)]

    def close(self):
        self.rt.synchronize()
        for name in ("candidates","mask","keep","result","counters","count"):
            getattr(self,name).close()
