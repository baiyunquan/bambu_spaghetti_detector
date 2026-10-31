#!/usr/bin/env python3
"""Serial, offline correctness/benchmark/soak checks; never printer control."""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ["OPENBLAS_CORETYPE"]="ARMV8"
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"inference"))
import cv2
import numpy as np


def snapshot():
    result={}
    for line in Path("/proc/self/status").read_text().splitlines():
        key=line.split(":")[0]
        if key in ("VmRSS","VmHWM","VmSwap"):
            result[key]=int(line.split()[1])
    pss=0
    with open("/proc/self/smaps") as source:
        for line in source:
            if line.startswith("Pss:"):pss+=int(line.split()[1])
    result["Pss"]=pss
    for line in Path("/proc/meminfo").read_text().splitlines():
        key=line.split(":")[0]
        if key in ("MemAvailable","SwapFree"):result[key]=int(line.split()[1])
    return result


def no_torch():
    assert not any(x=="torch" or x.startswith("torch.") or x=="torchvision" or
                   x.startswith("torchvision.") for x in sys.modules), "Native imported tensor framework"
    maps=Path("/proc/self/maps").read_text()
    assert "libtorch" not in maps and "/torchvision/" not in maps, "Native mapped tensor framework"


def validate_soak(samples,final,seconds):
    stable=[row for row in samples if row["elapsed_s"]>=min(600.,seconds/4.)]
    assert len(stable)>=2,"Insufficient steady samples"
    baseline=stable[0]
    series=stable+[final]
    assert all(row["VmRSS"]<=int(1.5*1024*1024) for row in series),"Steady RSS gate"
    for key in ("VmRSS","Pss"):
        assert max(row[key] for row in series)-baseline[key]<=16*1024,key+" growth gate"
    assert max(row["VmSwap"] for row in series)<=baseline["VmSwap"],"Worker swap growth gate"
    assert min(row["SwapFree"] for row in series)>=baseline["SwapFree"],"System swap growth gate"
    return dict({key:final[key]-baseline[key] for key in ("VmRSS","Pss","VmSwap")},
                system_swap=baseline["SwapFree"]-final["SwapFree"])


def cpu_preprocess(payload,target):
    image=cv2.imdecode(np.frombuffer(payload,np.uint8),cv2.IMREAD_COLOR)
    if image is None:raise ValueError("Bad reference JPEG")
    h,w=image.shape[:2];th,tw=target
    scale=min(tw/w,th/h);nw,nh=int(round(w*scale)),int(round(h*scale))
    px,py=(tw-nw)/2.,(th-nh)/2.
    l,t=int(round(px-.1)),int(round(py-.1))
    resized=cv2.resize(image,(nw,nh),interpolation=cv2.INTER_LINEAR)
    padded=cv2.copyMakeBorder(resized,t,th-nh-t,l,tw-nw-l,cv2.BORDER_CONSTANT,value=(114,114,114))
    tensor=np.ascontiguousarray(padded[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)/np.float32(255.)
    return tensor,dict(orig_h=h,orig_w=w,target_h=th,target_w=tw,scale=scale,pad_x=px,pad_y=py)


def equivalent(want,got):
    if len(want)!=len(got):return False,"count {} != {}".format(len(want),len(got))
    remaining=list(got)
    for item in want:
        choices=[(i,r) for i,r in enumerate(remaining) if r["class_id"]==item["class_id"] and
                 r["label"]==item["label"] and abs(r["confidence"]-item["confidence"])<=.0010001 and
                 max(abs(a-b) for a,b in zip(r["bbox"],item["bbox"]))<=1]
        if not choices:return False,"no match for {}".format(item)
        i,_=min(choices,key=lambda pair:sum(abs(a-b) for a,b in zip(pair[1]["bbox"],item["bbox"])))
        remaining.pop(i)
    return True,""


def decision_trace(detections,threshold):
    from api.decision_engine import DecisionEngine
    logging.getLogger("DecisionEngine").setLevel(100)
    engine=DecisionEngine(consecutive_required=6,confidence_threshold=threshold,
                          target_classes=["spaghetti","warping"],dry_run=True,cooldown_seconds=0)
    return [(r["has_failure"],r["consecutive_hits"],r["triggered"])
            for r in [engine.evaluate_frame(detections,"RUNNING") for _ in range(8)]]


def equivalence(args,files,emit):
    from native_cuda.pipeline import NativePipeline
    from reference import decode
    pipe=NativePipeline(args.engine,args.model_type,args.conf,.45)
    cpu_buffer=None
    failures=[];max_pixel_error=0.
    try:
        pipe.warmup()
        no_torch()
        cpu_buffer=pipe.rt.empty(pipe.detector.input_shape)
        for index,path in enumerate(files):
            payload=path.read_bytes()
            got=pipe.detect(payload)
            gpu_input=pipe.rt.download(pipe.pre.tensor)
            cpu_input,meta=cpu_preprocess(payload,pipe.detector.get_input_resolution())
            error=float(np.abs(gpu_input-cpu_input).max())
            max_pixel_error=max(max_pixel_error,error)
            pipe.rt.upload(cpu_input,cpu_buffer)
            raw=pipe.detector.infer(cpu_buffer)
            arrays=[pipe.rt.download(tensor) for tensor in raw]
            want=decode(arrays,args.model_type,meta,args.conf,.45,pipe.detector.class_names)
            ok,reason=equivalent(want,got)
            decision_ok=decision_trace(want,args.conf)==decision_trace(got,args.conf)
            ok=ok and decision_ok and error<=1.0001/255.
            if not ok:
                failures.append(dict(image=path.name,reason=reason,pixel_error=error,
                                     decision_equal=decision_ok,reference=want,native=got))
            if index%25==0:
                emit(dict(stage="equivalence",index=index+1,total=len(files),
                          failures=len(failures),max_pixel_error=max_pixel_error,**snapshot()))
        no_torch()
        emit(dict(stage="equivalence_summary",frames=len(files),failures=failures,
                  max_pixel_error=max_pixel_error,**snapshot()))
        return not failures
    finally:
        if cpu_buffer is not None:cpu_buffer.close()
        pipe.close()


def replay(args,files,emit):
    from worker_server import create_pipeline
    started=time.monotonic()
    pipe=create_pipeline(args.engine,args.model_type,args.conf,.45,args.backend)
    try:
        pipe.warmup()
        startup=time.monotonic()-started
        assert startup<=120.,"Worker startup deadline exceeded"
        emit(dict(stage="ready",startup_s=startup,backend=args.backend,**snapshot()))
        if args.backend=="native_cuda":no_torch()
        frame=0;times=[];snapshots=[];next_sample=0.
        begin=time.monotonic()
        duration=args.seconds if args.phase=="soak" else float("inf")
        while time.monotonic()-begin<duration and (args.phase=="soak" or frame<args.frames):
            payload=files[frame%len(files)].read_bytes()
            start=time.perf_counter()
            result=pipe.detect(payload)
            elapsed=(time.perf_counter()-start)*1000.
            times.append(elapsed);frame+=1
            now=time.monotonic()-begin
            if now>=next_sample:
                row=dict(stage="sample",elapsed_s=now,frames=frame,last_ms=elapsed,
                         detections=len(result),**snapshot())
                snapshots.append(row);emit(row);next_sample=now+30.
                if args.backend=="native_cuda":no_torch()
            if args.phase=="soak":
                time.sleep(max(0.,.5-(time.perf_counter()-start)))
        final=snapshot()
        growth=validate_soak(snapshots,final,args.seconds) if args.phase=="soak" else {}
        emit(dict(stage="replay_summary",backend=args.backend,phase=args.phase,
                  duration_s=time.monotonic()-begin,frames=frame,first_ms=times[0],
                  median_ms=float(np.median(times[10:] or times)),
                  p95_ms=float(np.percentile(times[10:] or times,95)),growth_kib=growth,**final))
        if args.backend=="native_cuda":
            no_torch()
            assert final["VmHWM"]<=int(1.8*1024*1024),"Peak RSS gate"
            assert final["VmRSS"]<=int(1.5*1024*1024),"Steady RSS gate"
            assert times[0]<5000.,"First request gate"
        if args.phase=="soak":
            assert time.monotonic()-begin>=args.seconds,"Soak ended early"
        return True
    finally:pipe.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine",required=True)
    parser.add_argument("--model-type",required=True,choices=["obico","yolov8","yolov5"])
    parser.add_argument("--backend",default="native_cuda",choices=["native_cuda","torch"])
    parser.add_argument("--phase",required=True,choices=["equivalence","benchmark","soak"])
    parser.add_argument("--dataset",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--conf",type=float,default=.35)
    parser.add_argument("--seconds",type=float,default=3600.)
    parser.add_argument("--frames",type=int,default=100)
    parser.add_argument("--camera",help="First benchmark/soak image, captured from the real A1")
    args=parser.parse_args()
    root=Path(args.dataset)
    files=sorted(p for p in root.iterdir() if p.suffix.lower() in (".jpg",".jpeg",".png"))
    if not files:parser.error("No replay images")
    if args.camera and args.phase!="equivalence":
        files=[Path(args.camera)]+files
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("w") as log:
        def emit(row):
            row["model_type"]=args.model_type
            line=json.dumps(row,allow_nan=False)
            log.write(line+"\n");log.flush()
            if row["stage"]!="equivalence_summary":
                print(line,flush=True)
        if args.phase=="equivalence":ok=equivalence(args,files,emit)
        else:ok=replay(args,files,emit)
    return 0 if ok else 1


if __name__=="__main__":sys.exit(main())
