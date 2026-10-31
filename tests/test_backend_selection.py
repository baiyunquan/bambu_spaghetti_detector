"""Backend routing and UDS errors without CUDA, printer access, or torch."""
import importlib
import json
from pathlib import Path
import socket
import struct
import subprocess
import sys
from types import SimpleNamespace
import threading
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'inference'))
sys.path.insert(0, str(ROOT))


def worker():
    return importlib.import_module('worker_server')


def test_worker_module_import_never_imports_torch():
    code = '''import builtins, sys
sys.path.insert(0, %r)
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'torchvision'):
        raise AssertionError('worker eagerly imported ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import worker_server
assert 'torch' not in sys.modules
''' % str(ROOT / 'inference')
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_default_and_explicit_torch_use_legacy_gpu_components():
    calls = []
    class Detector:
        def __init__(self, path, model_type):
            calls.append(('detector', path, model_type))
            self.class_names = ['spaghetti']
        def get_input_resolution(self): return (416, 640)
        def infer(self, tensor): return ['raw']
    class Preprocessor:
        def __init__(self, target_size): calls.append(('pre', target_size))
        def decode_and_preprocess(self, jpeg): return 'tensor', {'frame': jpeg}
    class Postprocessor:
        def __init__(self, conf_thresh, iou_thresh): calls.append(('post', conf_thresh, iou_thresh))
        def process_obico(self, raw, meta, labels):
            assert raw == ['raw'] and meta == {'frame': b'jpeg'} and labels == ['spaghetti']
            return [{'label': 'spaghetti'}]
    modules = {'trt_detector': SimpleNamespace(TensorRTDetector=Detector),
               'gpu_preprocess': SimpleNamespace(GPUPreprocessor=Preprocessor),
               'postprocess': SimpleNamespace(GPUPostProcessor=Postprocessor)}
    with patch.dict(sys.modules, modules):
        for kwargs in ({}, {'backend': 'torch'}):
            pipeline = worker().create_pipeline('weights.engine', 'obico', .3, .4, **kwargs)
            assert pipeline.detect(b'jpeg') == [{'label': 'spaghetti'}]
            pipeline.close()
    assert calls == [('detector', 'weights.engine', 'obico'), ('pre', (416, 640)), ('post', .3, .4)] * 2


def test_native_dispatch_and_failure_never_falls_back():
    calls = []
    sentinel = object()
    def native(*args):
        calls.append(args)
        return sentinel
    with patch.dict(sys.modules, {'native_cuda.pipeline': SimpleNamespace(NativePipeline=native)}):
        assert worker().create_pipeline('model', 'yolov5', .3, .4, backend='native_cuda') is sentinel
    assert calls == [('model', 'yolov5', .3, .4)]
    def broken(*args): raise RuntimeError('native initialization failed')
    with patch.dict(sys.modules, {'native_cuda.pipeline': SimpleNamespace(NativePipeline=broken)}):
        with pytest.raises(RuntimeError, match='native initialization failed'):
            worker().create_pipeline('model', 'yolov5', .3, .4, backend='native_cuda')
    with pytest.raises(ValueError, match='backend'):
        worker().create_pipeline('model', 'obico', .3, .4, backend='invalid')


def test_cli_defaults_and_explicit_backend_forwarded():
    for extra, expected in [([], 'torch'), (['--backend', 'native_cuda'], 'native_cuda')]:
        with patch.object(sys, 'argv', ['worker', '--engine', 'model'] + extra), patch.object(worker(), 'run_worker') as run:
            worker().main()
        assert run.call_args.kwargs['backend'] == expected


def exchange(payload, fail=False, declared_size=None):
    left, right = socket.socketpair()
    right.settimeout(2)
    class Pipeline:
        def detect(self, jpeg):
            if fail: raise ValueError('bad JPEG')
            return [{'label': 'spaghetti', 'confidence': .9, 'bbox': [1, 2, 3, 4]}]
    def serve():
        try: worker().serve_connection(left, Pipeline())
        finally: left.close()
    thread = threading.Thread(target=serve)
    thread.start()
    try:
        right.sendall(struct.pack('>I', len(payload) if declared_size is None else declared_size) + payload)
        size = struct.unpack('>I', worker().recv_exact(right, 4))[0]
        data = json.loads(worker().recv_exact(right, size).decode())
    finally:
        right.close()
        thread.join(2)
    assert not thread.is_alive()
    return data


def test_uds_success_and_inference_error_response():
    response = exchange(b'jpeg')
    assert response['status'] == 'ok' and response['detections'][0]['label'] == 'spaghetti'
    assert response['inference_time_ms'] >= 0
    response = exchange(b'broken', fail=True)
    assert response == {'status': 'error', 'error': 'bad JPEG', 'detections': []}


@pytest.mark.parametrize('size', [0, 5 * 1024 * 1024 + 1])
def test_uds_rejects_invalid_length_before_reading_body(size):
    response = exchange(b'', declared_size=size)
    assert response['status'] == 'error' and response['detections'] == []


def test_worker_error_raises_client_and_closes_connection():
    from api.ipc_client import InferenceClient
    left, right = socket.socketpair()
    client = InferenceClient()
    client.sock = right
    response = json.dumps({'status': 'error', 'error': 'inference unavailable', 'detections': []}).encode()
    left.sendall(struct.pack('>I', len(response)) + response)
    try:
        with pytest.raises(RuntimeError, match='inference unavailable'):
            client.detect(b'jpeg')
        assert client.sock is None
    finally:
        left.close()
        client.close()


def test_active_socket_and_regular_file_preserved(tmp_path):
    path = str(tmp_path / 'worker.sock')
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    server.listen(1)
    try:
        with pytest.raises(RuntimeError, match='active'):
            worker().remove_stale_socket(path)
        assert Path(path).exists()
    finally: server.close()
    worker().remove_stale_socket(path)
    assert not Path(path).exists()
    Path(path).write_text('important')
    with pytest.raises(RuntimeError):
        worker().remove_stale_socket(path)
    assert Path(path).read_text() == 'important'


def test_native_warmup_precedes_bind_and_cleanup_on_bind_failure(tmp_path):
    # Native CUDA is the external boundary; exercise the real worker lifecycle
    # against an occupied local endpoint rather than mocking the server socket.
    path = str(tmp_path / 'worker.sock')
    events = []
    class Pipeline:
        def __init__(self, *args): events.append('create')
        def warmup(self):
            assert not Path(path).exists()
            events.append('warmup')
            Path(path).write_text('occupied-after-warmup')
        def close(self): events.append('close')
    with patch.dict(sys.modules, {'native_cuda.pipeline': SimpleNamespace(NativePipeline=Pipeline)}), patch.object(worker(), 'SOCKET_PATH', path):
        with pytest.raises(RuntimeError):
            worker().run_worker('model', backend='native_cuda')
    assert events == ['create', 'warmup', 'close']
    assert Path(path).read_text() == 'occupied-after-warmup'


def test_native_warmup_failure_closes_pipeline(tmp_path):
    events = []
    class Pipeline:
        def __init__(self, *args): pass
        def warmup(self): raise ValueError('warmup failed')
        def close(self): events.append('close')
    with patch.dict(sys.modules, {'native_cuda.pipeline': SimpleNamespace(NativePipeline=Pipeline)}), patch.object(worker(), 'SOCKET_PATH', str(tmp_path / 'none.sock')):
        with pytest.raises(ValueError, match='warmup failed'):
            worker().run_worker('model', backend='native_cuda')
    assert events == ['close']


@pytest.mark.parametrize('configured,expected', [(None, 'torch'), ('native_cuda', 'native_cuda')])
def test_controller_forwards_backend_to_worker(configured, expected):
    import main
    app = main.SpaghettiDetectionApp.__new__(main.SpaghettiDetectionApp)
    detection = {'model_engine': 'models/test.engine'}
    if configured is not None:
        detection['backend'] = configured
    app.config = {'detection': detection}
    app.worker_process = None
    process = SimpleNamespace(poll=lambda: 1, returncode=1)
    with patch.object(main.os.path, 'exists', return_value=False), patch.object(Path, 'exists', return_value=True), patch.object(main.subprocess, 'Popen', return_value=process) as spawn:
        with pytest.raises(RuntimeError, match='startup'):
            app._ensure_worker_running()
    command = spawn.call_args.args[0]
    assert command[command.index('--backend') + 1] == expected


def test_nonfinite_inference_output_returns_error():
    left, right = socket.socketpair()
    right.settimeout(2)
    class Pipeline:
        def detect(self, payload): return [{'confidence': float('nan')}]
    def serve():
        try: worker().serve_connection(left, Pipeline())
        finally: left.close()
    thread = threading.Thread(target=serve)
    thread.start()
    try:
        right.sendall(struct.pack('>I', 1) + b'x')
        length = struct.unpack('>I', worker().recv_exact(right, 4))[0]
        response = json.loads(worker().recv_exact(right, length))
        assert response['status'] == 'error'
        assert response['detections'] == []
    finally:
        right.close()
        thread.join(2)


def test_controller_rejects_invalid_backend_before_socket_reuse():
    import main
    app = main.SpaghettiDetectionApp.__new__(main.SpaghettiDetectionApp)
    app.config = {'detection': {'backend': 'unknown'}}
    with patch.object(main.os.path, 'exists', return_value=True), patch.object(main.socket, 'socket'):
        with pytest.raises(ValueError, match='backend'):
            app._ensure_worker_running()


def test_actual_bind_error_closes_server_and_native_pipeline(tmp_path):
    events = []
    class Pipeline:
        def __init__(self, *args): pass
        def warmup(self): events.append('warmup')
        def close(self): events.append('pipeline close')
    class Server:
        def bind(self, path):
            events.append('bind')
            raise OSError('bind failed')
        def close(self): events.append('socket close')
    with patch.dict(sys.modules, {'native_cuda.pipeline': SimpleNamespace(NativePipeline=Pipeline)}), patch.object(worker(), 'SOCKET_PATH', str(tmp_path / 'absent.sock')), patch.object(worker().socket, 'socket', return_value=Server()):
        with pytest.raises(OSError, match='bind failed'):
            worker().run_worker('model', backend='native_cuda')
    assert events == ['warmup', 'bind', 'socket close', 'pipeline close']


def test_controller_leaves_stale_path_cleanup_to_worker(tmp_path):
    import main
    path = tmp_path / 'unrelated.sock'
    path.write_text('preserve me')
    app = main.SpaghettiDetectionApp.__new__(main.SpaghettiDetectionApp)
    app.config = {'detection': {}}
    app.worker_process = None
    process = SimpleNamespace(poll=lambda: 1, returncode=1)
    with patch.object(main, 'SOCKET_PATH', str(path)), patch.object(Path, 'exists', return_value=True), patch.object(main.subprocess, 'Popen', return_value=process):
        with pytest.raises(RuntimeError, match='startup'):
            app._ensure_worker_running()
    assert path.read_text() == 'preserve me'
