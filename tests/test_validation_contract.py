import importlib.util
import sys
import unittest
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
spec=importlib.util.spec_from_file_location("native_validation",str(ROOT/"scripts/validate_native.py"))
validation=importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


class TestValidationContract(unittest.TestCase):
    def test_preprocessing_reference_retains_rgb_planes(self):
        image=np.array([[[0,0,255],[0,255,0]],[[255,0,0],[255,255,255]]],np.uint8)
        _,data=cv2.imencode(".png",image)
        tensor,meta=validation.cpu_preprocess(data.tobytes(),(2,2))
        self.assertTrue(tensor.flags.c_contiguous)
        self.assertEqual(tensor.reshape(-1).tolist(),
                         [1.,0.,0.,1.,0.,1.,0.,1.,0.,0.,1.,1.])
        self.assertEqual(meta["scale"],1.)

    def test_result_comparison_checks_class_and_tolerances(self):
        want=[dict(label="spaghetti",class_id=3,confidence=.8,bbox=[1,2,10,20])]
        close=[dict(label="spaghetti",class_id=3,confidence=.8009,bbox=[2,2,10,20])]
        self.assertTrue(validation.equivalent(want,close)[0])
        close[0]["class_id"]=0
        self.assertFalse(validation.equivalent(want,close)[0])
        self.assertFalse(validation.equivalent(want,[])[0])

    def test_error_label_enters_requested_failure_trace(self):
        error=[dict(label="error",class_id=0,confidence=.99,bbox=[1,2,10,20])]
        trace=validation.decision_trace(error,.35)
        self.assertFalse(trace[4][2])
        self.assertTrue(trace[5][2])
        failure=[dict(label="spaghetti",class_id=3,confidence=.99,bbox=[1,2,10,20])]
        trace=validation.decision_trace(failure,.35)
        self.assertFalse(trace[4][2])
        self.assertTrue(trace[5][2])


if __name__=="__main__":unittest.main()
