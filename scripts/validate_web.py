#!/usr/bin/env python3
"""Short offline, single-UDS-client UI/inference comparison; no printer APIs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def memory(pid):
    values={}
    for row in Path('/proc/{}/status'.format(pid)).read_text().splitlines():
        key=row.split(':')[0]
        if key in ('VmRSS','VmHWM','VmSwap'):values[key]=int(row.split()[1])
    return values


def percentile(values,percent):
    values=sorted(values);index=(len(values)-1)*percent/100
    lo=int(index);hi=min(lo+1,len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(index-lo)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--frame',required=True)
    parser.add_argument('--photos',required=True)
    parser.add_argument('--web',action='store_true')
    parser.add_argument('--frames',type=int,default=100)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    from main import SpaghettiDetectionApp
    from api.ipc_client import InferenceClient,SOCKET_PATH
    assert not os.path.lexists(SOCKET_PATH),'Do not share another controller UDS connection'
    app=SpaghettiDetectionApp(args.config)
    app.config['web']={'enabled':args.web,'host':'0.0.0.0','port':8000}
    app.config['storage']={'snapshots_dir':args.photos}
    client=InferenceClient();times=[];stop=threading.Event();thread=None;errors=[]
    payload=Path(args.frame).read_bytes();started=time.monotonic()
    def get(path):
        with urllib.request.urlopen('http://127.0.0.1:8000'+path,timeout=3) as response:
            return response.read()
    def readers():
        while not stop.wait(2):
            try:
                get('/api/jetson');get('/api/jetson');get('/photos')
            except Exception as exc:errors.append(str(exc));return
    try:
        app._start_web()
        app._ensure_worker_running()
        startup=time.monotonic()-started
        assert startup<=120
        if args.web:
            assert app.web_server and app.web_server.is_alive
            for path in ['/photos','/jetson','/static/vendor/bootstrap.min.css',
                         '/static/vendor/chart.umd.min.js','/static/app.css','/static/jetson.js']:get(path)
            from urllib.parse import quote
            paths=['/media/'+quote(p.name) for p in sorted(Path(args.photos).glob('*.jpg'))[:24]]
            with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(get,paths))
            metrics=json.loads(get('/api/jetson'))
            assert metrics['status']=='ok' and metrics['current']['ram_total_mb']>0
            tegra_pid=app.web_server.sampler.pid
            thread=threading.Thread(target=readers,daemon=True);thread.start()
        else:tegra_pid=None
        client.connect()
        for i in range(args.frames):
            start=time.perf_counter();client.detect(payload);times.append((time.perf_counter()-start)*1000)
        assert not errors,errors
        assert all(t<5000 for t in times)
        assert not any(k=='torch' or k.startswith('torch.') or k=='torchvision' or k=='tensorrt' for k in sys.modules)
        controller=memory(os.getpid());tegra=memory(tegra_pid) if tegra_pid else {'VmRSS':0,'VmHWM':0,'VmSwap':0}
        worker=memory(app.worker_process.pid)
        assert worker['VmRSS']<=1.5*1024*1024
        assert worker['VmSwap']==0
        if args.web:assert app.web_server.sampler.pid==tegra_pid
        report={'web':args.web,'frames':len(times),'frame_sha256':hashlib.sha256(payload).hexdigest(),
                'engine':app.config['detection']['model_engine'],'confidence':app.config['detection']['confidence_threshold'],
                'startup_s':startup,'first_ms':times[0],'p95_ms':percentile(times[10:],95),
                'times_ms':times,'controller_kib':controller,'tegrastats_kib':tegra,'worker_kib':worker,
                'no_gpu_imports_in_controller':True,'http_errors':errors}
    finally:
        stop.set()
        if thread is not None:thread.join(timeout=5)
        client.close()
        worker_process=app.worker_process
        shutdown_start=time.monotonic()
        app.stop()
        print(json.dumps({'shutdown_s':time.monotonic()-shutdown_start,
                          'worker_exit_code':worker_process.returncode if worker_process else None,
                          'socket_removed':not os.path.lexists(SOCKET_PATH)}),flush=True)
    assert not os.path.lexists(SOCKET_PATH)
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='times_ms'}),flush=True)


if __name__=='__main__':main()
