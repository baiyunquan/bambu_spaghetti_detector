"""An inference error is a gap in consecutive evidence, never a valid hit."""
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from api.decision_engine import DecisionEngine

HIT = [{'label': 'spaghetti', 'confidence': .9, 'bbox': [1, 1, 5, 5]}]


def test_invalid_frame_breaks_streak_without_advancing_valid_statistics(caplog):
    engine = DecisionEngine(consecutive_required=3)
    engine.evaluate_frame(HIT, 'RUNNING')
    engine.evaluate_frame(HIT, 'RUNNING')
    assert engine.consecutive_hits == 2
    engine.invalidate_frame()
    assert engine.consecutive_hits == 0
    assert engine.total_frames_processed == 2
    assert engine.total_failures_detected == 2
    assert engine.last_trigger_time == 0
    result = engine.evaluate_frame(HIT, 'RUNNING')
    assert result['consecutive_hits'] == 1
    assert not result['triggered']
    assert 'invalid' in caplog.text.lower() and '2' in caplog.text


def test_main_inference_error_cannot_bridge_positive_frames(tmp_path, capsys, caplog):
    import main
    config = tmp_path / 'config.json'
    snapshots = tmp_path / 'snapshots'
    config.write_text(json.dumps({
        'printer': {'access_code': '12345678'},
        'detection': {'consecutive_hits': 3},
        'safety': {'dry_run': True},
        'storage': {'snapshots_dir': str(snapshots)},
    }))
    encoded = io.BytesIO()
    Image.new('RGB', (8, 8)).save(encoded, format='JPEG')
    jpeg = encoded.getvalue()

    class MQTT:
        gcode_state = 'RUNNING'
        current_layer = 2
        progress_percent = 10
        def __init__(self, **kwargs): pass
        def start(self): pass
        def stop(self): pass
        def pause_print(self): raise AssertionError('No printer control allowed')
        def stop_print(self): raise AssertionError('No printer control allowed')

    class Video:
        def __init__(self, **kwargs): pass
        def stream_frames(self): return iter([jpeg] * 4)
        def disconnect(self): pass

    class IPC:
        def __init__(self): self.index = 0
        def connect(self): pass
        def close(self): pass
        def detect(self, frame):
            self.index += 1
            if self.index == 3:
                raise RuntimeError('worker returned status error')
            return HIT

    app = main.SpaghettiDetectionApp(str(config))
    with patch.object(main.SpaghettiDetectionApp, '_ensure_worker_running'), patch.object(main.signal, 'signal'), patch.object(main, 'A1MQTTController', MQTT), patch.object(main, 'A1VideoStream', Video), patch.object(main, 'InferenceClient', IPC):
        app.run()
    output = capsys.readouterr().out
    fourth_frame = output.split('[Frame #4]')[-1]
    assert '[Hits: 1/3]' in fourth_frame
    assert 'CONFIRMED FAILURE' not in caplog.text
    assert not list(snapshots.glob('failure_*'))
    assert len(list(snapshots.glob('print_*.jpg'))) == 4
    records = [json.loads(p.read_text()) for p in snapshots.glob('print_*.json')]
    assert sum(bool(r['inference_error']) for r in records) == 1
