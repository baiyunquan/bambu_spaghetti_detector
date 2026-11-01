import json
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
from unittest.mock import patch
import urllib.request

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def controller(tmp_path,web=True):
    import main
    path=tmp_path/'config.json'
    config={'web':{'enabled':web,'host':'127.0.0.1','port':8000},
            'storage':{'snapshots_dir':str(tmp_path/'photos')},
            'printer':{'access_code':'fake','target_serial':'039TEST'},'safety':{'dry_run':True}}
    path.write_text(json.dumps(config))
    app=main.SpaghettiDetectionApp(str(path))
    assert hasattr(app,'_start_web'), 'Controller web lifecycle not implemented'
    return main,app


def fake_web(events,fail=False):
    class Web:
        def __init__(self,**kwargs):self.kwargs=kwargs
        def start(self):
            events.append('web.start')
            if fail:raise RuntimeError('web failed')
        def stop(self):events.append('web.stop')
    return SimpleNamespace(EmbeddedWebServer=Web)


def test_web_starts_before_gpu_and_closes_on_early_failure(tmp_path):
    main,app=controller(tmp_path);events=[]
    def worker():events.append('worker');raise RuntimeError('worker failed')
    with patch.dict(sys.modules,{'web.server':fake_web(events)}),patch.object(app,'_ensure_worker_running',worker),patch.object(main.signal,'signal'):
        with pytest.raises(RuntimeError,match='worker failed'):app.run()
    assert events==['web.start','worker','web.stop']
    app.stop();assert events.count('web.stop')==1


def test_web_failure_does_not_skip_inference_startup(tmp_path):
    main,app=controller(tmp_path);events=[]
    def worker():events.append('worker');raise RuntimeError('worker deliberately stopped')
    with patch.dict(sys.modules,{'web.server':fake_web(events,fail=True)}),patch.object(app,'_ensure_worker_running',worker),patch.object(main.signal,'signal'):
        with pytest.raises(RuntimeError,match='worker deliberately stopped'):app.run()
    assert events==['web.start','web.stop','worker']


def test_old_configuration_does_not_import_web_runtime(tmp_path):
    main,app=controller(tmp_path)
    del app.config['web']
    with patch.dict(sys.modules,{'web.server':None}):app._start_web()
    assert app.web_server is None


def test_partial_monitoring_startup_closes_every_owned_component(tmp_path):
    main,app=controller(tmp_path);events=[]
    class MQTT:
        def __init__(self,**kw):pass
        def start(self):events.append('mqtt.start')
        def stop(self):events.append('mqtt.stop')
    class IPC:
        def connect(self):raise RuntimeError('ipc failed')
        def close(self):events.append('ipc.close')
    with patch.dict(sys.modules,{'web.server':fake_web(events)}),patch.object(app,'_ensure_worker_running'),patch.object(main,'A1MQTTController',MQTT),patch.object(main,'InferenceClient',IPC),patch.object(main.signal,'signal'):
        with pytest.raises(RuntimeError,match='ipc failed'):app.run()
    app.stop()
    assert events.count('mqtt.stop')==events.count('ipc.close')==events.count('web.stop')==1


def test_normal_monitoring_exit_closes_all_components_once(tmp_path):
    main,app=controller(tmp_path);events=[]
    class MQTT:
        def __init__(self,**kw):pass
        def start(self):pass
        def stop(self):events.append('mqtt.stop')
    class IPC:
        def connect(self):pass
        def close(self):events.append('ipc.close')
    class Video:
        def __init__(self,**kw):pass
        def stream_frames(self):return iter([])
        def disconnect(self):events.append('video.disconnect')
    with patch.dict(sys.modules,{'web.server':fake_web(events)}),patch.object(app,'_ensure_worker_running'),patch.object(main,'A1MQTTController',MQTT),patch.object(main,'InferenceClient',IPC),patch.object(main,'A1VideoStream',Video),patch.object(main.signal,'signal'):
        app.run()
    app.stop()
    for event in ['mqtt.stop','ipc.close','video.disconnect','web.stop']:
        assert events.count(event)==1


class Sampler:
    def __init__(self):self.running=False;self.starts=0
    def start(self):self.running=True;self.starts+=1
    def stop(self):self.running=False
    def snapshot(self):return {'status':'unavailable','samples':[],'current':None}


def test_embedded_server_is_single_and_stops(tmp_path):
    assert (ROOT/'web/server.py').exists(), 'Embedded server not implemented'
    from web.server import EmbeddedWebServer
    sampler=Sampler();server=EmbeddedWebServer(tmp_path,host='127.0.0.1',port=0,sampler=sampler)
    server.start()
    try:
        server.start()
        url='http://127.0.0.1:{}/api/jetson'.format(server.port)
        assert json.load(urllib.request.urlopen(url,timeout=2))['status']=='unavailable'
        assert sampler.starts==1
    finally:server.stop();server.stop()
    assert not sampler.running and not server.is_alive
    with pytest.raises(OSError):urllib.request.urlopen(url,timeout=.2)


def test_port_conflict_cleans_lifespan_without_exiting_controller(tmp_path):
    assert (ROOT/'web/server.py').exists(), 'Embedded server not implemented'
    from web.server import EmbeddedWebServer
    sampler=Sampler()
    with socket.socket() as occupied:
        occupied.bind(('127.0.0.1',0));occupied.listen()
        server=EmbeddedWebServer(tmp_path,host='127.0.0.1',port=occupied.getsockname()[1],sampler=sampler)
        with pytest.raises(RuntimeError):server.start()
        server.stop()
    assert not sampler.running and not server.is_alive


def test_controller_stop_waits_for_owned_worker_to_exit(tmp_path):
    import subprocess
    main,app=controller(tmp_path,web=False)
    child=subprocess.Popen([sys.executable,'-u','-c',
        'import signal,sys,time;signal.signal(signal.SIGTERM,lambda *a:(time.sleep(.2),sys.exit(0)));print("ready",flush=True);time.sleep(60)'],
        stdout=subprocess.PIPE,text=True,start_new_session=True)
    try:
        assert child.stdout.readline().strip()=='ready'
        app.worker_process=child
        app.stop()
        assert child.poll()==0,'Controller returned before its worker exited'
    finally:
        if child.poll() is None:child.terminate();child.wait(timeout=3)
        child.stdout.close()
