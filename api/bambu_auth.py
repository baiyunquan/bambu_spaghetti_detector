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
        """Authenticates with Bambu Cloud and caches token."""
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

        logger.info(f"Connecting to Bambu Cloud ({self.region})...")
        resp = requests.post(url, json=payload, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if data.get("success"):
            token = data.get("accessToken")
            self._save_token(token)
            logger.info("Authentication successful!")
            return token

        if data.get("loginType") == "verifyCode":
            logger.warning("Bambu Cloud requested 2FA Email verification.")
            requests.post(f"{self.base_url}/v1/user-service/user/sendemail", json={"email": username, "type": "login"}, headers=headers)
            code = input("Enter 6-digit verification code sent to your email: ").strip()
            v_payload = {"account": username, "code": code}
            v_resp = requests.post(url, json=v_payload, headers=headers, timeout=15)
            v_data = v_resp.json()
            if v_data.get("success"):
                token = v_data.get("accessToken")
                self._save_token(token)
                logger.info("2FA Verification successful!")
                return token
            raise RuntimeError(f"2FA verification failed: {v_data}")

        raise RuntimeError(f"Login failed: {data.get('message', data)}")

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
