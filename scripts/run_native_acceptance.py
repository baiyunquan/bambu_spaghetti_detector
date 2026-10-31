#!/usr/bin/env python3
"""Offline serial acceptance driver, including three real one-hour soaks."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda:source.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def summary(path):
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    return rows[-1]


def freeze_inputs(paths):
    return {str(path):sha(path) for path in paths}


def verify_inputs(inputs):
    for path,checksum in inputs.items():
        assert sha(Path(path))==checksum,"Inputs changed during acceptance: "+path


def require_corpus(directory,manifest_path):
    files=sorted(p for p in directory.iterdir() if p.suffix.lower() in (".jpg",".jpeg",".png"))
    if len(files)!=295:raise ValueError("Acceptance requires exactly 295 held-out images")
    manifest=json.loads(Path(manifest_path).read_text())
    actual={p.name:sha(p) for p in files}
    if manifest.get("count")!=295 or manifest.get("files")!=actual:
        raise ValueError("Corpus differs from the approved held-out inventory")
    return files


def require_equivalence(row):
    assert row.get("stage")=="equivalence_summary" and row.get("frames")==295,"Incomplete equivalence corpus"
    assert row.get("failures")==[],"Equivalence failures"
    assert row.get("max_pixel_error",0.)<=1.0001/255.,"Preprocessing error gate"


def require_uds_result(text):
    rows=[json.loads(line) for line in text.splitlines() if line.startswith("{")]
    ready=[row for row in rows if row.get("stage")=="ready"]
    frames=[row for row in rows if row.get("stage")=="frame"]
    assert len(ready)==1 and ready[0]["startup_s"]<=120.,"UDS startup gate"
    assert ready[0]["client_timeout_s"]==5.,"UDS default timeout changed"
    assert len(frames)==5 and "[PASS]" in text,"Incomplete/skipped UDS test"
    assert all(row["response"]["status"]=="ok" and row["roundtrip_ms"]<5000. for row in frames),"UDS frame gate"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models",required=True)
    parser.add_argument("--dataset",required=True)
    parser.add_argument("--logs",required=True)
    parser.add_argument("--python",default="/home/jetson/Spaghetti_Failure_Detection/inference/.venv-jetpack/bin/python")
    parser.add_argument("--phase",choices=["all","functional","benchmark","soak"],default="all")
    parser.add_argument("--corpus-manifest",required=True)
    parser.add_argument("--camera",default="/home/jetson/spaghetti-diagnostics-20261031/camera.jpg")
    parser.add_argument("--controller-python",default="/home/jetson/Spaghetti_Failure_Detection/api/.venv/bin/python")
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    models=Path(args.models);logs=Path(args.logs)
    logs.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env["OPENBLAS_CORETYPE"]="ARMV8"
    variants=[("obico","obico_spaghetti",.35),
              ("yolov8","yolov8_failure_416",.35),("yolov8","yolov8_failure_640",.35),
              ("yolov5","yolov5_failure_416",.30),("yolov5","yolov5_failure_640",.30)]
    stress=[variants[0],variants[2],variants[4]]
    files=require_corpus(Path(args.dataset),args.corpus_manifest)
    paths=[path for path in sorted(root.rglob("*.py"))
           if "__pycache__" not in str(path) and ".venv" not in str(path)]
    paths+=list((root/"inference/native_cuda").glob("*.cu"))
    paths+=[root/"inference/native_cuda/build/kernels.ptx",
            root/"inference/native_cuda/build/kernels.json",
            Path(args.corpus_manifest),Path(args.camera)]+files
    inputs=freeze_inputs(paths)
    for _,name,_ in variants:
        for suffix in (".engine",".engine.meta.json"):
            path=models/(name+suffix)
            inputs[str(path)]=sha(path)
    (logs/"inputs.json").write_text(json.dumps(inputs,indent=2))

    def run(tag,command,timeout=1800):
        outfile=logs/(tag+".stdout.log")
        with outfile.open("w") as output:
            child=subprocess.Popen(command,env=env,stdout=output,stderr=subprocess.STDOUT)
            started=time.monotonic();next_status=0.
            try:
                while child.poll() is None:
                    elapsed=time.monotonic()-started
                    if elapsed>timeout:raise TimeoutError(tag+" deadline")
                    if elapsed>=next_status:
                        print(json.dumps(dict(task=tag,elapsed_s=round(elapsed,1),pid=child.pid)),flush=True)
                        next_status=elapsed+30.
                    time.sleep(1.)
                if child.returncode:
                    print(outfile.read_text()[-10000:],flush=True)
                    raise RuntimeError("{} exited {} (see {})".format(tag,child.returncode,outfile))
            finally:
                if child.poll() is None:
                    child.terminate()
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
        print(json.dumps(dict(task=tag,status="passed",duration_s=round(time.monotonic()-started,1))),flush=True)
        return outfile

    def check(kind,name,conf,phase,backend="native_cuda"):
        tag="{}-{}-{}".format(phase,name,backend)
        output=logs/(tag+".jsonl")
        command=[args.python,str(root/"scripts/validate_native.py"),
                 "--engine",str(models/(name+".engine")),"--model-type",kind,
                 "--dataset",args.dataset,"--output",str(output),"--conf",str(conf),
                 "--phase",phase,"--backend",backend,"--camera",args.camera]
        run(tag,command,3900 if phase=="soak" else 1800)
        row=summary(output)
        if phase=="equivalence":require_equivalence(row)
        return row

    results={"phase":args.phase,"functional":False,"benchmarks":{},"soaks":{}}
    if args.phase in ("all","functional"):
        gpu_env=dict(env);gpu_env["SPAGHETTI_NATIVE_GPU_TEST"]="1"
        old=env
        env=gpu_env
        run("gpu-contracts",[args.python,str(root/"tests/test_native_gpu.py"),"-v"])
        env=old
        for kind,name,conf in variants:
            check(kind,name,conf,"equivalence")
            saved=env
            env=dict(env,SPAGHETTI_TEST_ENGINE=str(models/(name+".engine")),
                     SPAGHETTI_TEST_TYPE=kind,SPAGHETTI_TEST_CONF=str(conf),
                     SPAGHETTI_TEST_BACKEND="native_cuda",SPAGHETTI_TEST_FRAME=args.camera)
            try:
                path=run("uds-"+name,[args.controller_python,str(root/"tests/test_e2e_inference.py")],180)
                require_uds_result(path.read_text())
            finally:env=saved
        results["functional"]=True
    if args.phase in ("all","benchmark"):
        # Include the currently deployed 416 YOLOv5 engine as well as the
        # larger variants used for resource stress.
        for kind,name,conf in stress+[variants[3]]:
            old=check(kind,name,conf,"benchmark","torch")
            new=check(kind,name,conf,"benchmark")
            saving=old["VmRSS"]-new["VmRSS"]
            ratio=new["p95_ms"]/old["p95_ms"]
            assert saving>=800*1024,"RSS saving gate: "+name
            assert ratio<=1.10,"P95 regression gate: "+name
            results["benchmarks"][name]=dict(saving_kib=saving,p95_ratio=ratio,legacy=old,native=new)
    if args.phase in ("all","soak"):
        for kind,name,conf in stress:
            results["soaks"][name]=check(kind,name,conf,"soak")
    verify_inputs(inputs)
    results["all_passed"]=(results["functional"] and len(results["benchmarks"])==4 and len(results["soaks"])==3)
    (logs/"result.json").write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps(results),flush=True)
    return 0


if __name__=="__main__":sys.exit(main())
