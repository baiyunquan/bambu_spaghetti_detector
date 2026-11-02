"""
Bambu Lab A1 Local MQTT Controller (Port 8883)
==============================================
Monitors printing telemetry, maintains the printer state machine, and issues
pause/stop control commands over secure local MQTT.
"""

import ssl
import json
import time
import logging
import threading
from typing import Optional, Callable, Dict, Any
import paho.mqtt.client as mqtt

logger = logging.getLogger("MQTTController")

STAGE_DESCRIPTIONS = {
    0: "正在打印",
    1: "自动热床调平中",
    2: "热床预热中",
    3: "机械振动校准中",
    4: "正在换料",
    5: "暂停 (M400)",
    6: "暂停 (耗材耗尽)",
    7: "喷嘴加热中",
    8: "挤出流量校准中",
    9: "扫描热床表面",
    10: "首层打印检测中",
    11: "识别构建板类型",
    12: "激光雷达校准中",
    13: "工具头复位归零中",
    14: "清洁喷嘴中",
    15: "检查挤出机温度",
    16: "用户手动暂停",
    17: "工具头前盖脱落暂停",
    18: "激光雷达校准中",
    19: "动态挤出流量校准",
    20: "喷嘴温度异常暂停",
    21: "热床温度异常暂停",
    22: "耗材退料中",
    23: "步进电机丢步暂停",
    24: "耗材进料中",
    25: "电机静音校准中",
    26: "AMS通讯丢失暂停",
    27: "喉管风扇转速过低暂停",
    28: "机箱温度异常暂停",
    29: "机箱降温中",
    30: "用户代码暂停",
    32: "喷嘴缠料检测暂停",
    33: "切刀故障暂停",
    34: "第一层缺陷暂停",
    35: "喷嘴堵头暂停",
    255: "空闲就绪",
}

SPEED_PROFILES = {
    1: "静音 (50%)",
    2: "标准 (100%)",
    3: "运动 (124%)",
    4: "狂暴 (166%)",
}


class A1MQTTController:
    def __init__(self, printer_ip: str, access_code: str, dev_id: str, port: int = 8883):
        self.printer_ip = printer_ip
        self.access_code = access_code
        self.dev_id = dev_id
        self.port = port
        self._lock = threading.Lock()

        # Connectivity & Uptime
        self.is_connected = False
        self.connected_since: Optional[float] = None
        self.last_message_time: Optional[float] = None

        # Print state & Progress
        self.gcode_state: str = "UNKNOWN"
        self.current_layer: int = 0
        self.total_layers: int = 0
        self.progress_percent: int = 0
        self.remaining_time_min: int = 0
        self.subtask_name: str = ""
        self.gcode_file: str = ""
        self.stage_code: int = 255
        self.stage_text: str = "空闲就绪"
        self.speed_level: int = 2
        self.speed_text: str = "标准 (100%)"
        self.speed_magnitude: int = 100

        # Thermal Telemetry
        self.nozzle_temper: float = 0.0
        self.nozzle_target_temper: float = 0.0
        self.bed_temper: float = 0.0
        self.bed_target_temper: float = 0.0
        self.chamber_temper: float = 0.0

        # Fans
        self.cooling_fan_speed: int = 0
        self.heatbreak_fan_speed: int = 0
        self.aux_fan_speed: int = 0

        # Hardware & Connectivity
        self.wifi_signal: str = "--"
        self.sdcard: bool = False
        self.active_tray: Dict[str, Any] = {}

        client_id = f"spaghetti_detector_{int(time.time())}"
        if hasattr(mqtt, "CallbackAPIVersion"):
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id=client_id)
        else:
            self.client = mqtt.Client(client_id=client_id)
        self._setup_client()

    def _setup_client(self):
        self.client.username_pw_set("bblp", self.access_code)
        self.client.tls_set(cert_reqs=ssl.CERT_NONE)
        self.client.tls_insecure_set(True)

        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info("Connected to A1 local MQTT broker!")
            with self._lock:
                self.is_connected = True
                if self.connected_since is None:
                    self.connected_since = time.time()
            topic = f"device/{self.dev_id}/report"
            self.client.subscribe(topic)
            logger.info(f"Subscribed to topic: {topic}")
            self.request_pushall()
        else:
            logger.error(f"Failed to connect to A1 MQTT broker (rc={rc})")
            with self._lock:
                self.is_connected = False
                self.connected_since = None

    def _on_disconnect(self, client, userdata, rc):
        logger.warning(f"Disconnected from A1 MQTT broker (rc={rc})")
        with self._lock:
            self.is_connected = False
            self.connected_since = None

    def _to_float(self, value, default=0.0):
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def _to_int(self, value, default=0):
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return default

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            if "print" in payload:
                p_data = payload["print"]
                with self._lock:
                    self.last_message_time = time.time()
                    if not self.is_connected:
                        self.is_connected = True
                        if self.connected_since is None:
                            self.connected_since = time.time()

                    # Print progress & state
                    if "gcode_state" in p_data:
                        self.gcode_state = str(p_data["gcode_state"]).upper()
                    if "layer_num" in p_data:
                        self.current_layer = self._to_int(p_data["layer_num"], self.current_layer)
                    if "total_layer_num" in p_data:
                        self.total_layers = self._to_int(p_data["total_layer_num"], self.total_layers)
                    if "mc_percent" in p_data:
                        self.progress_percent = self._to_int(p_data["mc_percent"], self.progress_percent)
                    if "mc_remaining_time" in p_data:
                        self.remaining_time_min = self._to_int(p_data["mc_remaining_time"], self.remaining_time_min)
                    if "subtask_name" in p_data and p_data["subtask_name"]:
                        self.subtask_name = str(p_data["subtask_name"])
                    if "gcode_file" in p_data and p_data["gcode_file"]:
                        self.gcode_file = str(p_data["gcode_file"])

                    # Stages & Speed
                    stage_id = p_data.get("stg_cur")
                    if stage_id is None and isinstance(p_data.get("stage"), dict):
                        stage_id = p_data.get("stage", {}).get("_id")
                    if stage_id is not None:
                        self.stage_code = self._to_int(stage_id, 255)
                        self.stage_text = STAGE_DESCRIPTIONS.get(self.stage_code, f"阶段 #{self.stage_code}")

                    if "spd_lvl" in p_data:
                        self.speed_level = self._to_int(p_data["spd_lvl"], self.speed_level)
                        self.speed_text = SPEED_PROFILES.get(self.speed_level, f"挡位 {self.speed_level}")
                    if "spd_mag" in p_data:
                        self.speed_magnitude = self._to_int(p_data["spd_mag"], self.speed_magnitude)

                    # Temperatures
                    if "nozzle_temper" in p_data:
                        self.nozzle_temper = self._to_float(p_data["nozzle_temper"], self.nozzle_temper)
                    if "nozzle_target_temper" in p_data:
                        self.nozzle_target_temper = self._to_float(p_data["nozzle_target_temper"], self.nozzle_target_temper)
                    if "bed_temper" in p_data:
                        self.bed_temper = self._to_float(p_data["bed_temper"], self.bed_temper)
                    if "bed_target_temper" in p_data:
                        self.bed_target_temper = self._to_float(p_data["bed_target_temper"], self.bed_target_temper)
                    if "chamber_temper" in p_data:
                        self.chamber_temper = self._to_float(p_data["chamber_temper"], self.chamber_temper)

                    # Fans
                    if "cooling_fan_speed" in p_data:
                        self.cooling_fan_speed = self._to_int(p_data["cooling_fan_speed"], self.cooling_fan_speed)
                    if "heatbreak_fan_speed" in p_data:
                        self.heatbreak_fan_speed = self._to_int(p_data["heatbreak_fan_speed"], self.heatbreak_fan_speed)
                    if "big_fan1_speed" in p_data:
                        self.aux_fan_speed = self._to_int(p_data["big_fan1_speed"], self.aux_fan_speed)

                    # Hardware
                    if "wifi_signal" in p_data:
                        self.wifi_signal = str(p_data["wifi_signal"])
                    if "sdcard" in p_data:
                        self.sdcard = bool(p_data["sdcard"])

                    # Filament / Tray
                    tray_now = p_data.get("tray_now")
                    if "vt_tray" in p_data and isinstance(p_data["vt_tray"], dict):
                        vt = p_data["vt_tray"]
                        raw_color = vt.get("tray_color") or (vt.get("cols", [""])[0] if vt.get("cols") else "")
                        color_hex = f"#{raw_color[:6]}" if raw_color and len(raw_color) >= 6 else "#CCCCCC"
                        self.active_tray = {
                            "type": vt.get("tray_type", "PLA"),
                            "color": color_hex,
                            "remain_percent": self._to_int(vt.get("remain", 0), 0),
                            "name": "外部料架 (External Spool)"
                        }
                    elif "ams" in p_data and isinstance(p_data["ams"], dict) and "ams" in p_data["ams"]:
                        ams_list = p_data["ams"].get("ams", [])
                        if ams_list and isinstance(ams_list, list) and "tray" in ams_list[0]:
                            trays = ams_list[0].get("tray", [])
                            idx = self._to_int(tray_now, 0) % 4 if tray_now is not None else 0
                            if idx < len(trays):
                                tr = trays[idx]
                                raw_color = tr.get("tray_color") or (tr.get("cols", [""])[0] if tr.get("cols") else "")
                                color_hex = f"#{raw_color[:6]}" if raw_color and len(raw_color) >= 6 else "#CCCCCC"
                                self.active_tray = {
                                    "type": tr.get("tray_type", "PLA"),
                                    "color": color_hex,
                                    "remain_percent": self._to_int(tr.get("remain", 0), 0),
                                    "name": f"AMS 料槽 {idx + 1}"
                                }

        except Exception as e:
            logger.debug(f"Failed to parse MQTT message: {e}")

    def get_telemetry_snapshot(self) -> Dict[str, Any]:
        """Thread-safe snapshot of all current printer parameters."""
        with self._lock:
            now = time.time()
            uptime_seconds = int(now - self.connected_since) if (self.is_connected and self.connected_since) else 0
            hours = uptime_seconds // 3600
            minutes = (uptime_seconds % 3600) // 60
            uptime_formatted = f"{hours}小时{minutes}分" if hours > 0 else f"{minutes}分"

            rem_h = self.remaining_time_min // 60
            rem_m = self.remaining_time_min % 60
            rem_formatted = f"{rem_h}小时{rem_m}分" if rem_h > 0 else f"{rem_m}分"

            return {
                "online": self.is_connected,
                "uptime_seconds": uptime_seconds,
                "uptime_formatted": uptime_formatted,
                "printer_ip": self.printer_ip,
                "dev_id": self.dev_id,
                "last_message_at": self.last_message_time,
                "gcode_state": self.gcode_state,
                "current_layer": self.current_layer,
                "total_layers": self.total_layers,
                "progress_percent": self.progress_percent,
                "remaining_time_min": self.remaining_time_min,
                "remaining_time_formatted": rem_formatted,
                "subtask_name": self.subtask_name or "无活动任务",
                "gcode_file": self.gcode_file,
                "stage_code": self.stage_code,
                "stage_text": self.stage_text,
                "speed_level": self.speed_level,
                "speed_text": self.speed_text,
                "speed_magnitude": self.speed_magnitude,
                "temperatures": {
                    "nozzle": self.nozzle_temper,
                    "nozzle_target": self.nozzle_target_temper,
                    "bed": self.bed_temper,
                    "bed_target": self.bed_target_temper,
                    "chamber": self.chamber_temper,
                },
                "fans": {
                    "cooling_fan_speed": self.cooling_fan_speed,
                    "heatbreak_fan_speed": self.heatbreak_fan_speed,
                    "aux_fan_speed": self.aux_fan_speed,
                },
                "hardware": {
                    "wifi_signal": self.wifi_signal,
                    "sdcard": self.sdcard,
                },
                "active_tray": self.active_tray or {
                    "type": "PLA",
                    "color": "#10b981",
                    "remain_percent": 100,
                    "name": "料槽 1"
                },
            }

    def start(self):
        """Starts background MQTT network loop."""
        logger.info(f"Starting MQTT client for {self.printer_ip}:{self.port}...")
        self.client.connect_async(self.printer_ip, self.port, keepalive=30)
        self.client.loop_start()

    def stop(self):
        """Stops MQTT client loop and disconnects."""
        try:
            self.client.disconnect()
        except Exception:
            pass
        self.client.loop_stop()
        with self._lock:
            self.is_connected = False
            self.connected_since = None

    def send_command(self, command_dict: dict) -> bool:
        """Publishes control command to printer request topic."""
        topic = f"device/{self.dev_id}/request"
        payload_str = json.dumps(command_dict)
        logger.info(f"Publishing command to {topic}: {payload_str}")
        result = self.client.publish(topic, payload_str, qos=1)
        return result.rc == mqtt.MQTT_ERR_SUCCESS

    def pause_print(self) -> bool:
        """Sends immediate pause command to A1."""
        cmd = {"print": {"sequence_id": "0", "command": "pause"}}
        return self.send_command(cmd)

    def stop_print(self) -> bool:
        """Sends stop / abort print command to A1."""
        cmd = {"print": {"sequence_id": "0", "command": "stop"}}
        return self.send_command(cmd)

    def resume_print(self) -> bool:
        """Sends resume print command to A1."""
        cmd = {"print": {"sequence_id": "0", "command": "resume"}}
        return self.send_command(cmd)

    def request_pushall(self) -> bool:
        """Requests complete state dump from printer."""
        cmd = {"pushing": {"sequence_id": "0", "command": "pushall"}}
        return self.send_command(cmd)

