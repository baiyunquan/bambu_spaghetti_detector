import json
import time
import sys
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.mqtt_controller import A1MQTTController, STAGE_DESCRIPTIONS, SPEED_PROFILES


def test_mqtt_telemetry_snapshot_offline_defaults():
    ctrl = A1MQTTController(printer_ip="192.168.199.231", access_code="12345678", dev_id="03900D6A1521014")
    snapshot = ctrl.get_telemetry_snapshot()
    assert snapshot["online"] is False
    assert snapshot["uptime_seconds"] == 0
    assert snapshot["printer_ip"] == "192.168.199.231"
    assert snapshot["dev_id"] == "03900D6A1521014"
    assert snapshot["gcode_state"] == "UNKNOWN"
    assert snapshot["temperatures"]["nozzle"] == 0.0
    assert snapshot["temperatures"]["bed"] == 0.0


def test_mqtt_telemetry_message_parsing():
    ctrl = A1MQTTController(printer_ip="192.168.199.231", access_code="12345678", dev_id="03900D6A1521014")
    # Simulate connection
    ctrl._on_connect(None, None, None, 0)
    assert ctrl.is_connected is True
    assert ctrl.connected_since is not None

    sample_report = {
        "print": {
            "gcode_state": "RUNNING",
            "layer_num": 42,
            "total_layer_num": 180,
            "mc_percent": 23,
            "mc_remaining_time": 75,
            "subtask_name": "Benchy_3D_Model",
            "gcode_file": "/sdcard/Benchy_3D_Model.gcode",
            "stg_cur": 0,
            "spd_lvl": 3,
            "spd_mag": 124,
            "nozzle_temper": 215.5,
            "nozzle_target_temper": 220.0,
            "bed_temper": 60.0,
            "bed_target_temper": 60.0,
            "chamber_temper": 28.0,
            "cooling_fan_speed": "80",
            "heatbreak_fan_speed": 100,
            "big_fan1_speed": 50,
            "wifi_signal": "-48dBm",
            "sdcard": True,
            "vt_tray": {
                "tray_type": "PETG",
                "tray_color": "FF5500FF",
                "remain": 85
            }
        }
    }
    fake_msg = MagicMock()
    fake_msg.payload = json.dumps(sample_report).encode("utf-8")
    ctrl._on_message(None, None, fake_msg)

    snap = ctrl.get_telemetry_snapshot()
    assert snap["online"] is True
    assert snap["gcode_state"] == "RUNNING"
    assert snap["current_layer"] == 42
    assert snap["total_layers"] == 180
    assert snap["progress_percent"] == 23
    assert snap["remaining_time_min"] == 75
    assert snap["remaining_time_formatted"] == "1小时15分"
    assert snap["subtask_name"] == "Benchy_3D_Model"
    assert snap["stage_code"] == 0
    assert snap["stage_text"] == "正在打印"
    assert snap["speed_level"] == 3
    assert snap["speed_text"] == "运动 (124%)"
    assert snap["speed_magnitude"] == 124
    assert snap["temperatures"]["nozzle"] == 215.5
    assert snap["temperatures"]["nozzle_target"] == 220.0
    assert snap["temperatures"]["bed"] == 60.0
    assert snap["temperatures"]["bed_target"] == 60.0
    assert snap["temperatures"]["chamber"] == 28.0
    assert snap["fans"]["cooling_fan_speed"] == 80
    assert snap["fans"]["heatbreak_fan_speed"] == 100
    assert snap["fans"]["aux_fan_speed"] == 50
    assert snap["hardware"]["wifi_signal"] == "-48dBm"
    assert snap["hardware"]["sdcard"] is True
    assert snap["active_tray"]["type"] == "PETG"
    assert snap["active_tray"]["color"] == "#FF5500"
    assert snap["active_tray"]["remain_percent"] == 85
