"""Run explicitly on Nano with SPAGHETTI_NATIVE_GPU_TEST=1."""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"inference"))
ENABLED=os.environ.get("SPAGHETTI_NATIVE_GPU_TEST")=="1"


@unittest.skipUnless(ENABLED,"Explicit Jetson GPU test")
class TestNativeGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from native_cuda.runtime import Runtime
        cls.rt=Runtime()
        cls.stream=cls.rt.stream

    @classmethod
    def tearDownClass(cls):
        cls.rt.close()

    def test_preprocessing_matches_rgb_reference(self):
        import cv2
        import numpy as np
        from native_cuda.preprocess import GPUPreprocessor
        for h,w in [(2,2),(37,53),(720,1280),(1080,1536)]:
            with self.subTest(size=(w,h)):
                image=np.random.RandomState(42).randint(0,256,(h,w,3),dtype=np.uint8)
                _,encoded=cv2.imencode(".png",image)
                pre=GPUPreprocessor(self.rt,(416,416),self.stream)
                try:
                    tensor,meta=pre.decode_and_preprocess(encoded.tobytes())
                    got=self.rt.download(tensor)
                    scale=meta["scale"];nw,nh=int(round(w*scale)),int(round(h*scale))
                    left,top=int(round(meta["pad_x"]-.1)),int(round(meta["pad_y"]-.1))
                    resized=cv2.resize(image,(nw,nh),interpolation=cv2.INTER_LINEAR)
                    padded=cv2.copyMakeBorder(resized,top,416-nh-top,left,416-nw-left,
                                             cv2.BORDER_CONSTANT,value=(114,114,114))
                    want=padded[:,:,::-1].transpose(2,0,1)[None].astype(np.float32)/255.
                    self.assertLessEqual(float(np.abs(got-want).max()),1.0001/255.)
                finally: pre.close()

    def test_preprocessing_avoids_opencv_cuda_module_initialization(self):
        import cv2
        import numpy as np
        from unittest.mock import patch
        from native_cuda.preprocess import GPUPreprocessor
        _, encoded=cv2.imencode(".png",np.zeros((37,53,3),np.uint8))
        with patch.object(cv2,"cuda_GpuMat",side_effect=AssertionError("OpenCV CUDA allocation")):
            pre=GPUPreprocessor(self.rt,(416,416),self.stream)
            try:
                tensor,_=pre.decode_and_preprocess(encoded.tobytes())
                self.assertEqual(self.rt.download(tensor).shape,(1,3,416,416))
            finally:pre.close()

    def _post(self,kind,arrays,classes,threshold=.35,iou=.45):
        from native_cuda.postprocess import GPUPostProcessor
        n=arrays[0].shape[2] if kind=="yolov8" else arrays[0].shape[1]
        post=GPUPostProcessor(self.rt,n,classes,kind,threshold,iou)
        tensors=[self.rt.upload(a) for a in arrays]
        try:
            return post.process(tensors,dict(target_h=640,target_w=640,orig_h=640,orig_w=640,
                                             scale=1.,pad_x=0.,pad_y=0.))
        finally:
            post.close()
            for t in tensors:t.close()

    def test_yolov5_objectness_class_and_ties(self):
        import numpy as np
        arrays=[np.array([[[100,100,40,40,1,0,0,0,.9],
                           [101,101,40,40,1,0,0,0,.9],
                           [100,100,40,40,1,.8,0,0,0],
                           [300,300,40,40,.1,0,0,0,.99]]],np.float32)]
        result=self._post("yolov5",arrays,["error","extrusor","part","spaghetti"])
        self.assertEqual([r["class_id"] for r in result],[3,0])
        self.assertEqual(result[0]["bbox"],[80,80,120,120])
        self.assertEqual(result[1]["label"],"error")

    def test_more_than_4096_candidates_are_not_truncated(self):
        import numpy as np
        n=25200
        out=np.zeros((1,n,9),np.float32)
        out[0,:,0]=np.arange(n)%200*3+1
        out[0,:,1]=np.arange(n)//200*3+1
        out[0,:,2:4]=1
        out[0,:,4]=1
        out[0,:,8]=.8
        result=self._post("yolov5",[out],["error","extrusor","part","spaghetti"])
        self.assertEqual(len(result),n)
        self.assertTrue(all(r["class_id"]==3 for r in result))

    def test_score_boundary_empty_and_nonfinite(self):
        import numpy as np
        arrays=[np.array([[[[0,0,.1,.1]]]],np.float32),np.array([[[.35]]],np.float32)]
        self.assertEqual(len(self._post("obico",arrays,["spaghetti"])),1)
        arrays[1][0,0,0]=.34
        self.assertEqual(self._post("obico",arrays,["spaghetti"]),[])
        arrays[1][0,0,0]=np.nan
        with self.assertRaises(ValueError):self._post("obico",arrays,["spaghetti"])

    def test_yolov8_matches_reference(self):
        import numpy as np
        from reference import decode
        rng=np.random.RandomState(5)
        out=rng.rand(1,6,321).astype(np.float32)
        out[:,:2,:]*=500
        out[:,2:4,:]*=40
        meta=dict(target_h=640,target_w=640,orig_h=640,orig_w=640,scale=1.,pad_x=0.,pad_y=0.)
        want=decode([out],"yolov8",meta,.35,.45,["spaghetti","warping"])
        got=self._post("yolov8",[out],["spaghetti","warping"])
        self.assertEqual(got,want)

    def test_iou_boundary_is_strict(self):
        import numpy as np
        arrays=[np.array([[[[0,0,1,1]],[[0,0,.5,1]]]],np.float32),
                np.array([[[.9],[.8]]],np.float32)]
        self.assertEqual(len(self._post("obico",arrays,["spaghetti"],iou=.5)),2)
        below=float(np.nextafter(np.float32(.5),np.float32(0)))
        self.assertEqual(len(self._post("obico",arrays,["spaghetti"],iou=below)),1)


if __name__=="__main__":unittest.main()
