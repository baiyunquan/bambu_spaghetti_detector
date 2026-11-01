import io
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
import pytest


def components(tmp_path):
    from api.snapshot import SnapshotManager
    from api.frame_recording import CaptureRequests, FrameRecorder
    requests = CaptureRequests()
    requests.start()
    return requests, FrameRecorder(SnapshotManager(str(tmp_path)), requests)


def jpeg():
    buffer = io.BytesIO()
    Image.new('RGB', (160, 100), 'white').save(buffer, 'JPEG')
    return buffer.getvalue()


def test_every_running_frame_is_saved_without_overwriting(tmp_path):
    requests, recorder = components(tmp_path)
    detections = [{'label': 'part', 'class_id': 2, 'confidence': .9, 'bbox': [50, 50, 90, 90]}]
    for state in ['RUNNING', 'RUNNING', 'FINISH', 'PAUSE', 'UNKNOWN']:
        recorder.record(jpeg(), detections, 0, state, {})
    images = list(tmp_path.glob('*.jpg'))
    assert len(images) == 2
    for path in images:
        metadata = json.loads(path.with_suffix('.json').read_text())
        assert metadata['detections'] == detections
        assert metadata['capture_kind'] == 'print'
        assert metadata['inference_error'] is None
        with Image.open(path) as image:
            red, green, blue = image.getpixel((50, 70))
            assert red > green + 40 and red > blue + 40
    requests.stop()


def test_manual_capture_uses_next_frame_even_when_idle(tmp_path):
    requests, recorder = components(tmp_path)
    recorder.record(jpeg(), [], 0, 'FINISH', {})
    assert not list(tmp_path.glob('*.jpg'))
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(requests.request, 2)
        assert requests.pending_event.wait(1)
        path = recorder.record(jpeg(), [], 0, 'FINISH', {})
        assert pending.result(timeout=1) == path.name
    assert len(list(tmp_path.glob('*.jpg'))) == 1
    assert json.loads(path.with_suffix('.json').read_text())['capture_kind'] == 'manual'


def test_manual_request_during_print_does_not_duplicate_frame(tmp_path):
    requests, recorder = components(tmp_path)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(requests.request, 2)
        assert requests.pending_event.wait(1)
        path = recorder.record(jpeg(), [], 2, 'RUNNING', {})
        assert pending.result(timeout=1) == path.name
    assert len(list(tmp_path.glob('*.jpg'))) == 1
    assert json.loads(path.with_suffix('.json').read_text())['manual_requested'] is True


def test_capture_requests_are_bounded_cancelled_and_closed(tmp_path):
    from api.frame_recording import CaptureBusy, CaptureUnavailable
    requests, recorder = components(tmp_path)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(requests.request, 2)
        assert requests.pending_event.wait(1)
        with pytest.raises(CaptureBusy): requests.request(.01)
        requests.stop()
        with pytest.raises(CaptureUnavailable): pending.result(timeout=1)
    with pytest.raises(CaptureUnavailable): requests.request(.01)
    requests.start()
    with pytest.raises(TimeoutError): requests.request(.01)
    recorder.record(jpeg(), [], 0, 'FINISH', {})
    assert not list(tmp_path.glob('*.jpg')), 'Timed out requests must not capture later frames'


@pytest.mark.parametrize('message', ['GPU unavailable', ''])
def test_inference_failure_still_saves_print_image_and_marks_error(tmp_path, message):
    _, recorder = components(tmp_path)
    path = recorder.record(jpeg(), [], 0, 'RUNNING', {}, inference_error=message)
    metadata = json.loads(path.with_suffix('.json').read_text())
    assert metadata['inference_error'] == (message or 'Inference failed')
    from web.app import create_app
    from fastapi.testclient import TestClient
    from test_web_gallery import Sampler
    with TestClient(create_app(tmp_path, sampler=Sampler())) as browser:
        detail = browser.get('/photos/' + path.name)
        assert '推理失败' in detail.text
        assert '无检测项' not in detail.text


def test_disk_failure_does_not_stop_monitoring_and_notifies_manual_request(tmp_path, monkeypatch):
    requests, recorder = components(tmp_path)
    def fail(**kwargs): raise OSError('disk full')
    monkeypatch.setattr(recorder.snapshots, 'save_frame', fail)
    assert recorder.record(jpeg(), [], 0, 'RUNNING', {}) is None
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(requests.request, 2)
        assert requests.pending_event.wait(1)
        assert recorder.record(jpeg(), [], 0, 'FINISH', {}) is None
        with pytest.raises(OSError): pending.result(timeout=1)


def test_manual_capture_does_not_use_frame_received_before_click(tmp_path):
    requests, recorder = components(tmp_path)
    received_before_click = time.monotonic()
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(requests.request, 2)
        assert requests.pending_event.wait(1)
        assert recorder.record(jpeg(), [], 0, 'FINISH', {},
                               frame_received_at=received_before_click) is None
        path = recorder.record(jpeg(), [], 0, 'FINISH', {}, frame_received_at=time.monotonic())
        assert pending.result(timeout=1) == path.name


@pytest.mark.parametrize('inference_fails', [False, True])
def test_controller_saves_print_stream_including_inference_errors(tmp_path, monkeypatch, inference_fails):
    import main
    from types import SimpleNamespace
    config = tmp_path/'config.json'
    config.write_text(json.dumps({'printer':{'access_code':'fake','target_serial':'039TEST'},
        'storage':{'snapshots_dir':str(tmp_path/'photos')}, 'safety':{'dry_run':True}}))
    app = main.SpaghettiDetectionApp(str(config)); app.running = True
    class IPC:
        def connect(self): pass
        def close(self): pass
        def detect(self, payload):
            if inference_fails: raise RuntimeError('GPU unavailable')
            return []
    monkeypatch.setattr(app, '_ensure_worker_running', lambda: None)
    monkeypatch.setattr(main, 'InferenceClient', IPC)
    monkeypatch.setattr(main, 'A1MQTTController', lambda **kw: SimpleNamespace(
        start=lambda:None,stop=lambda:None,gcode_state='RUNNING',current_layer=1,progress_percent=1))
    monkeypatch.setattr(main, 'A1VideoStream', lambda **kw: SimpleNamespace(
        stream_frames=lambda:iter([jpeg(),jpeg()]),disconnect=lambda:None))
    app._run_monitoring()
    records = list((tmp_path/'photos').glob('*.json'))
    assert len(records) == 2
    assert all(bool(json.loads(p.read_text())['inference_error']) == inference_fails for p in records)


def test_disk_full_does_not_prevent_confirmed_pause(tmp_path, monkeypatch):
    import main
    from types import SimpleNamespace
    config = tmp_path/'config.json'
    config.write_text(json.dumps({'printer':{'access_code':'fake','target_serial':'039TEST'},
        'storage':{'snapshots_dir':str(tmp_path/'photos')}, 'safety':{'dry_run':False}}))
    app = main.SpaghettiDetectionApp(str(config)); app.running = True
    commands = []
    class IPC:
        def connect(self): pass
        def close(self): pass
        def detect(self, payload):
            return [{'label':'spaghetti','confidence':.9,'bbox':[50,50,90,90],'class_id':3}]
    def disk_full(*args, **kwargs): raise OSError('No space left on device')
    monkeypatch.setattr(main.SnapshotManager, 'save_frame', disk_full)
    monkeypatch.setattr(app, '_ensure_worker_running', lambda: None)
    monkeypatch.setattr(main, 'InferenceClient', IPC)
    monkeypatch.setattr(main, 'A1MQTTController', lambda **kw: SimpleNamespace(
        start=lambda:None, stop=lambda:None, gcode_state='RUNNING',current_layer=1,progress_percent=1,
        pause_print=lambda:commands.append('pause'), stop_print=lambda:commands.append('stop')))
    monkeypatch.setattr(main, 'A1VideoStream', lambda **kw: SimpleNamespace(
        stream_frames=lambda:iter([jpeg() for _ in range(6)]),disconnect=lambda:None))
    app._run_monitoring()
    assert commands == ['pause']
