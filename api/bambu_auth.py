"""
Bambu Cloud Authentication & Device Discovery Module
===================================================
Authenticates with Bambu Cloud, handles token caching and 2FA, retrieves bound
printers, and strictly enforces serial number validation (prefix '039' for A1).
"""

import os
import json
import logging
from typing import Dict, Any, Optional
import requests

logger = logging.getLogger("BambuAuth")


class BambuAuth:
    def __init__(self, region: str = "china", token_file: str = "tokens.json"):
        self.region = region.lower()
        self.base_url = "https://api.bambulab.cn" if self.region == "china" else "https://api.bambulab.com"
        self.token_file = token_file
        self.token: Optional[str] = None
        self._load_token()

    def _load_token(self):
        if os.path.exists(self.token_file):
            try:
                with open(self.token_file, "r") as f:
                    data = json.load(f)
                    self.token = data.get("access_token")
            except Exception as e:
                logger.warning(f"Failed to load cached token: {e}")

    def _save_token(self, token: str):
        self.token = token
        try:
            with open(self.token_file, "w") as f:
                json.dump({"access_token": token, "region": self.region}, f, indent=2)
            os.chmod(self.token_file, 0o600)
        except Exception as e:
            logger.warning(f"Failed to save token to file: {e}")

    def login(self, username: str, password: str) -> str:
        """CLI synchronous login helper (caches token)."""
        res = self.login_step1(username, password)
        if res.get("status") == "success":
            return res["token"]
        if res.get("status") == "2fa_required":
            logger.warning("Bambu Cloud requested 2FA verification.")
            code = input("Enter verification code: ").strip()
            v_res = self.login_step2(username, code)
            if v_res.get("status") == "success":
                return v_res["token"]
            raise RuntimeError(v_res.get("message", "2FA verification failed"))
        raise RuntimeError(res.get("message", "Login failed"))

    def login_step1(self, username: str, password: str) -> Dict[str, Any]:
        """Non-blocking Step 1: authenticate or trigger email 2FA."""
        url = f"{self.base_url}/v1/user-service/user/login"
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "BambuStudio/01.09.03.50",
        }
        payload = {
            "account": username,
            "password": password,
            "apiError": ""
        }
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=15)
            data = resp.json()
        except Exception as e:
            return {"status": "error", "message": f"连接拓竹云端超时或网络错误: {e}"}

        token = data.get("accessToken")
        if token or data.get("success"):
            self._save_token(token)
            return {"status": "success", "token": token}

        if data.get("loginType") == "verifyCode":
            try:
                requests.post(f"{self.base_url}/v1/user-service/user/sendemail",
                              json={"email": username, "type": "login"}, headers=headers, timeout=10)
            except Exception:
                pass
            return {"status": "2fa_required", "message": "已向您的账号邮箱发送两步验证码，请输入以完成登录"}

        return {"status": "error", "message": data.get("message", f"登录失败: {data}")}

    def login_step2(self, username: str, code: str) -> Dict[str, Any]:
        """Non-blocking Step 2: verify 2FA code and obtain token."""
        url = f"{self.base_url}/v1/user-service/user/login"
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "BambuStudio/01.09.03.50",
        }
        payload = {
            "account": username,
            "code": code.strip(),
        }
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=15)
            data = resp.json()
        except Exception as e:
            return {"status": "error", "message": f"验证请求异常: {e}"}

        token = data.get("accessToken")
        if token or data.get("success"):
            self._save_token(token)
            return {"status": "success", "token": token}

        return {"status": "error", "message": data.get("message", "两步验证码无效或已过期")}

    def get_a1_credentials(self, target_serial: str = None) -> Dict[str, Any]:
        """
        Retrieves bound devices from cloud, validates target A1 printer,
        and extracts LAN Access Code.
        """
        if not self.token:
            raise RuntimeError("Not authenticated. Call login() or provide valid token.")

        headers = {
            "Authorization": f"Bearer {self.token}",
            "User-Agent": "BambuStudio/01.09.03.50",
        }
        url = f"{self.base_url}/v1/iot-service/api/user/bind"
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        devices = data.get("devices", [])
        if not devices:
            raise RuntimeError("No printers found in Bambu Cloud account.")

        selected_device = None
        for dev in devices:
            dev_id = dev.get("dev_id", "")
            # STRICT REQUIREMENT: Only accept A1 series with '039' prefix!
            # Never bind to P1S ('01P') or other models.
            if not dev_id.startswith("039"):
                logger.info(f"Skipping non-A1 printer: {dev.get('name')} (SN: {dev_id})")
                continue

            if target_serial and dev_id != target_serial:
                continue

            selected_device = dev
            break

        if not selected_device:
            raise RuntimeError(
                f"Target A1 printer (prefix '039') not found in cloud account. Bound devices: {[d.get('dev_id') for d in devices]}"
            )

        dev_id = selected_device["dev_id"]
        access_code = selected_device.get("dev_access_code", "")
        name = selected_device.get("name", "Bambu A1")

        if not access_code:
            raise RuntimeError(f"Cloud API did not return dev_access_code for printer {dev_id}.")

        logger.info(f"Target A1 Identified: {name} (SN: {dev_id})")
        return {
            "dev_id": dev_id,
            "access_code": access_code,
            "name": name,
            "online": selected_device.get("online", False),
            "model": selected_device.get("dev_product_name") or selected_device.get("dev_model_name", "A1")
        }


def test_lan_connectivity(ip: str, port: int = 8883, timeout: float = 3.0) -> bool:
    """Tests raw TCP connection to printer port (e.g. 8883 MQTT)."""
    import socket
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except OSError:
        return False


def update_config_printer(config_path: str, ip: str, access_code: str, target_serial: str) -> bool:
    """Updates printer credentials in config.json with strict serial prefix validation."""
    if not target_serial or not target_serial.startswith("039"):
        raise ValueError(f"安全约束拦截：仅允许连接拓竹 A1 打印机（序列号前缀必须为 039）。当前输入：{target_serial}")

    cfg = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            cfg = {}
    cfg.setdefault("printer", {})
    cfg["printer"]["printer_ip"] = ip.strip()
    cfg["printer"]["access_code"] = access_code.strip()
    cfg["printer"]["target_serial"] = target_serial.strip()

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    return True

