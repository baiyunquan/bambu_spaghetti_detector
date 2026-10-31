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
from typing import Optional, Callable
import paho.mqtt.client as mqtt

logger = logging.getLogger("MQTTController")


class A1MQTTController:
    def __init__(self, printer_ip: str, access_code: str, dev_id: str, port: int = 8883):
        self.printer_ip = printer_ip
        self.access_code = access_code
        self.dev_id = dev_id
        self.port = port

        # State tracking
        self.gcode_state: str = "UNKNOWN"  # RUNNING, PAUSE, IDLE, FINISH, etc.
        self.current_layer: int = 0
        self.progress_percent: int = 0
        self.is_connected = False

        self.client = mqtt.Client(client_id=f"spaghetti_detector_{int(time.time())}")
        self._setup_client()

    def _setup_client(self):
        self.client.username_pw_set("bblp", self.access_code)

        # Insecure TLS for local self-signed BBL CA
        self.client.tls_set(cert_reqs=ssl.CERT_NONE)
        self.client.tls_insecure_set(True)

        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info("Connected to A1 local MQTT broker!")
            self.is_connected = True
            # Subscribe to status report topic
            topic = f"device/{self.dev_id}/report"
            self.client.subscribe(topic)
            logger.info(f"Subscribed to topic: {topic}")
            # Request initial status push
            self.request_pushall()
        else:
            logger.error(f"Failed to connect to A1 MQTT broker (rc={rc})")
            self.is_connected = False

    def _on_disconnect(self, client, userdata, rc):
        logger.warning(f"Disconnected from A1 MQTT broker (rc={rc})")
        self.is_connected = False

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            if "print" in payload:
                p_data = payload["print"]
                if "gcode_state" in p_data:
                    self.gcode_state = p_data["gcode_state"]
                if "layer_num" in p_data:
                    self.current_layer = p_data["layer_num"]
                if "mc_percent" in p_data:
                    self.progress_percent = p_data["mc_percent"]
        except Exception as e:
            logger.debug(f"Failed to parse MQTT message: {e}")

    def start(self):
        """Starts background MQTT network loop."""
        logger.info(f"Starting MQTT client for {self.printer_ip}:{self.port}...")
        self.client.connect_async(self.printer_ip, self.port, keepalive=30)
        self.client.loop_start()

    def stop(self):
        """Stops MQTT client loop and disconnects."""
        self.client.loop_stop()
        self.client.disconnect()
        self.is_connected = False

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
