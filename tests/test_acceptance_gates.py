"""Acceptance failures must never be reported as a successful migration."""
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
def module(name,file):
    spec=importlib.util.spec_from_file_location(name,str(ROOT/"scripts"/file))
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result
validation=module("validation_gates","validate_native.py")
acceptance=module("acceptance_gates","run_native_acceptance.py")

def sample(elapsed=600):
    return dict(elapsed_s=elapsed,VmRSS=1024*1024,VmHWM=1200*1024,Pss=900*1024,
                VmSwap=0,SwapFree=2000*1024,MemAvailable=2200*1024)


class TestAcceptanceGates(unittest.TestCase):
    def test_startup_over_controller_budget_fails_actual_replay(self):
        import worker_server
        clock={"now":0.}
        class Pipeline:
            closed=False
            def warmup(self):clock["now"]=121.
            def detect(self,payload):return []
            def close(self):self.closed=True
        pipe=Pipeline()
        with TemporaryDirectory() as folder:
            image=Path(folder)/"frame.jpg";image.write_bytes(b"fixture")
            args=SimpleNamespace(engine="unused",model_type="obico",conf=.35,
                                 backend="native_cuda",phase="benchmark",frames=1,seconds=3600)
            with patch.object(worker_server,"create_pipeline",return_value=pipe),\
                 patch.object(validation.time,"monotonic",side_effect=lambda:clock["now"]),\
                 patch.object(validation,"snapshot",side_effect=lambda:{k:v for k,v in sample().items() if k!="elapsed_s"}),\
                 patch.object(validation,"no_torch"):
                with self.assertRaisesRegex(AssertionError,"startup"):
                    validation.replay(args,[image],lambda row:None)
        self.assertTrue(pipe.closed)

    def test_no_steady_samples_cannot_pass_actual_replay(self):
        import worker_server
        clock={"now":0.}
        class Pipeline:
            def warmup(self):pass
            def detect(self,payload):clock["now"]+=.02;return []
            def close(self):pass
        with TemporaryDirectory() as folder:
            image=Path(folder)/"frame.jpg";image.write_bytes(b"fixture")
            args=SimpleNamespace(engine="unused",model_type="obico",conf=.35,
                                 backend="native_cuda",phase="soak",frames=1,seconds=.01)
            with patch.object(worker_server,"create_pipeline",return_value=Pipeline()),\
                 patch.object(validation.time,"monotonic",side_effect=lambda:clock["now"]),\
                 patch.object(validation.time,"perf_counter",side_effect=lambda:clock["now"]),\
                 patch.object(validation.time,"sleep",side_effect=lambda n:clock.update(now=clock["now"]+n)),\
                 patch.object(validation,"snapshot",side_effect=lambda:{k:v for k,v in sample().items() if k!="elapsed_s"}),\
                 patch.object(validation,"no_torch"):
                with self.assertRaisesRegex(AssertionError,"steady samples"):
                    validation.replay(args,[image],lambda row:None)

    def test_middle_memory_violation_is_not_hidden_by_final_drop(self):
        rows=[sample(600),sample(630),sample(660)]
        rows[1]["VmRSS"]=1700*1024
        with self.assertRaises(AssertionError):
            validation.validate_soak(rows,sample(3600),3600)
        rows=[sample(600),sample(630),sample(660)]
        rows[1]["Pss"]+=17*1024
        with self.assertRaises(AssertionError):
            validation.validate_soak(rows,sample(3600),3600)

    def test_worker_or_system_swap_growth_fails(self):
        for field,delta in [("VmSwap",4),("SwapFree",-4)]:
            rows=[sample(600),sample(630)]
            rows[-1][field]+=delta
            with self.subTest(field=field):
                with self.assertRaises(AssertionError):
                    validation.validate_soak(rows,sample(3600),3600)

    def test_corpus_inventory_and_executed_binary_are_bound(self):
        with TemporaryDirectory() as folder:
            root=Path(folder)
            for i in range(294):(root/("{}.jpg".format(i))).write_bytes(b"image")
            manifest=root/"heldout.json"
            manifest.write_text(json.dumps({"count":295,"files":{}}))
            with self.assertRaises(ValueError):acceptance.require_corpus(root,manifest)
            image=root/"294.jpg";image.write_bytes(b"image")
            names={p.name:acceptance.sha(p) for p in root.glob("*.jpg")}
            manifest.write_text(json.dumps(dict(count=295,files=names)))
            files=acceptance.require_corpus(root,manifest)
            self.assertEqual(len(files),295)
            ptx=root/"kernels.ptx";ptx.write_bytes(b"PTX")
            frozen=acceptance.freeze_inputs([image,ptx])
            image.write_bytes(b"changed")
            with self.assertRaises(AssertionError):acceptance.verify_inputs(frozen)
            image.write_bytes(b"image");ptx.write_bytes(b"other PTX")
            with self.assertRaises(AssertionError):acceptance.verify_inputs(frozen)

    def test_incomplete_equivalence_and_skipped_uds_are_not_success(self):
        with self.assertRaises(AssertionError):
            acceptance.require_equivalence(dict(stage="equivalence_summary",frames=294,failures=[]))
        with self.assertRaises(AssertionError):
            acceptance.require_uds_result("[SKIP] Engine not found")


if __name__=="__main__":unittest.main()
