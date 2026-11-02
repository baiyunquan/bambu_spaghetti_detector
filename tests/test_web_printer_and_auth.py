import json
import sys
from pathlib import Path
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from web.app import create_app


class MockSampler:
    def start(self): pass
    def stop(self): pass
    def snapshot(self):
        return {'status': 'ok', 'current': {'cpu_percent': 15, 'gpu_percent': 10}, 'samples': []}


class MockMQTT:
    def __init__(self, connected=True):
        self.is_connected = connected
        self.printer_ip = "192.168.199.231"
        self.dev_id = "03900D6A1521014"

    def get_telemetry_snapshot(self):
        return {
            "online": self.is_connected,
            "uptime_seconds": 3600,
            "uptime_formatted": "1小时0分",
            "printer_ip": self.printer_ip,
            "dev_id": self.dev_id,
            "gcode_state": "RUNNING" if self.is_connected else "OFFLINE",
            "current_layer": 50,
            "total_layers": 200,
            "progress_percent": 25,
            "remaining_time_min": 60,
            "remaining_time_formatted": "1小时0分",
            "subtask_name": "Test_Model_Print",
            "gcode_file": "/sdcard/Test_Model.gcode",
            "stage_code": 0,
            "stage_text": "正在打印",
            "speed_level": 2,
            "speed_text": "标准 (100%)",
            "speed_magnitude": 100,
            "temperatures": {
                "nozzle": 220.0,
                "nozzle_target": 220.0,
                "bed": 60.0,
                "bed_target": 60.0,
                "chamber": 25.0
            },
            "fans": {
                "cooling_fan_speed": 100,
                "heatbreak_fan_speed": 100,
                "aux_fan_speed": 0
            },
            "hardware": {
                "wifi_signal": "-50dBm",
                "sdcard": True
            },
            "active_tray": {
                "type": "PLA",
                "color": "#10b981",
                "remain_percent": 90,
                "name": "料槽 1"
            }
        }


def test_home_redirect_when_connected_and_disconnected(tmp_path):
    # 1. When disconnected -> redirect to /login
    app_offline = create_app(tmp_path, sampler=MockSampler(), mqtt_controller=MockMQTT(connected=False))
    client_offline = TestClient(app_offline)
    resp = client_offline.get("/", follow_redirects=False)
    assert resp.status_code == 307
    assert resp.headers["location"] == "/login?reason=disconnected"

    # 2. When connected -> redirect to /status
    app_online = create_app(tmp_path, sampler=MockSampler(), mqtt_controller=MockMQTT(connected=True))
    client_online = TestClient(app_online)
    resp_online = client_online.get("/", follow_redirects=False)
    assert resp_online.status_code == 307
    assert resp_online.headers["location"] == "/status"


def test_status_page_and_telemetry_endpoint(tmp_path):
    mock_mqtt = MockMQTT(connected=True)
    app = create_app(tmp_path, sampler=MockSampler(), mqtt_controller=mock_mqtt)
    client = TestClient(app)

    # 1. Page render
    resp_html = client.get("/status")
    assert resp_html.status_code == 200
    assert "打印机状态" in resp_html.text
    assert "喷嘴温度" in resp_html.text

    # 2. JSON API
    resp_api = client.get("/api/printer")
    assert resp_api.status_code == 200
    data = resp_api.json()
    assert data["online"] is True
    assert data["subtask_name"] == "Test_Model_Print"
    assert data["temperatures"]["nozzle"] == 220.0
    assert data["fans"]["cooling_fan_speed"] == 100


def test_login_page_renders_with_config(tmp_path):
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({
        "printer": {"printer_ip": "192.168.199.231", "access_code": "87654321", "target_serial": "03900D6A1521014"},
        "bambu_cloud": {"account": "test@example.com", "region": "china"}
    }))

    app = create_app(tmp_path, sampler=MockSampler(), config_path=str(cfg_file))
    client = TestClient(app)

    resp = client.get("/login?reason=disconnected")
    assert resp.status_code == 200
    assert "局域网直接连接" in resp.text
    assert "拓竹云账号登录" in resp.text
    assert "192.168.199.231" in resp.text
    assert "test@example.com" in resp.text


def test_lan_login_saves_config_and_enforces_039_prefix(tmp_path):
    cfg_file = tmp_path / "config.json"
    reconnect_called = []

    def on_reconnect(ip, access_code, serial):
        reconnect_called.append((ip, access_code, serial))

    app = create_app(tmp_path, sampler=MockSampler(), config_path=str(cfg_file), on_reconnect=on_reconnect)
    client = TestClient(app)

    # 1. Reject non-039 prefix (e.g. P1S prefix '01P')
    bad_payload = {
        "ip": "192.168.199.164",
        "access_code": "12345678",
        "serial": "01P00C550801214"
    }
    bad_resp = client.post("/api/auth/lan-login", json=bad_payload)
    assert bad_resp.status_code == 400
    assert "039" in bad_resp.text
    assert len(reconnect_called) == 0

    # 2. Accept valid A1 serial starting with 039
    good_payload = {
        "ip": "192.168.199.231",
        "access_code": "21580921",
        "serial": "03900D6A1521014"
    }
    good_resp = client.post("/api/auth/lan-login", json=good_payload)
    assert good_resp.status_code == 200
    assert good_resp.json()["status"] == "ok"
    assert good_resp.json()["redirect"] == "/status"

    # Verify config saved
    saved_cfg = json.loads(cfg_file.read_text())
    assert saved_cfg["printer"]["printer_ip"] == "192.168.199.231"
    assert saved_cfg["printer"]["access_code"] == "21580921"
    assert saved_cfg["printer"]["target_serial"] == "03900D6A1521014"

    # Verify reconnect callback triggered
    assert len(reconnect_called) == 1
    assert reconnect_called[0] == ("192.168.199.231", "21580921", "03900D6A1521014")
