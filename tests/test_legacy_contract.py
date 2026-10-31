import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"inference"))
try:
    import torch
    import torchvision
    from postprocess import GPUPostProcessor
    AVAILABLE=True
except ImportError:
    AVAILABLE=False


@unittest.skipUnless(AVAILABLE,"Requires legacy tensor runtime")
class TestLegacyOutputContract(unittest.TestCase):
    def test_nonfinite_output_is_error_not_zero_detections(self):
        post=GPUPostProcessor(device="cpu")
        outputs=[torch.zeros((1,6,1))]
        outputs[0][0,4,0]=float("nan")
        with self.assertRaises(ValueError):
            post.process_yolov8(outputs,dict(target_w=640,target_h=640))

    def test_yolov5_keeps_original_spaghetti_class_id(self):
        post=GPUPostProcessor(device="cpu")
        out=torch.tensor([[[100.,100.,40.,40.,1.,0.,0.,0.,.9],
                           [200.,200.,40.,40.,.1,0.,0.,0.,.99]]])
        meta=dict(target_w=640,target_h=640,orig_w=640,orig_h=640,scale=1.,pad_x=0.,pad_y=0.)
        result=post.process_yolov5([out],meta)
        self.assertEqual([(r["label"],r["class_id"]) for r in result],[("spaghetti",3)])


if __name__=="__main__":unittest.main()
