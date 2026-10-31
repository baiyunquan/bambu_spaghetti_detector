#!/usr/bin/env python3
"""
Bambu Lab Access Code & Device Query Script
===========================================
Queries Bambu Lab Cloud to retrieve bound printers, their LAN Access Codes,
and device serial numbers.

Usage:
    python scripts/get_access_code.py
"""

import sys
import getpass
import requests

def main():
    print("=" * 60)
    print("      Bambu Lab Cloud - Device & Access Code Query")
    print("=" * 60)
    print("Region options: 'china' (Default for CN accounts) or 'global'")
    region = input("Region [china/global, default: china]: ").strip().lower() or "china"
    base_url = "https://api.bambulab.cn" if region == "china" else "https://api.bambulab.com"

    username = input("Bambu Cloud Email / Phone: ").strip()
    if not username:
        print("[ERROR] Email is required.")
        return 1

    password = getpass.getpass("Bambu Cloud Password: ").strip()
    if not password:
        print("[ERROR] Password is required.")
        return 1

    print("\n[INFO] Authenticating with Bambu Cloud...")
    session = requests.Session()
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "BambuStudio/01.09.03.50",
    }

    login_payload = {
        "account": username,
        "password": password,
        "apiError": ""
    }

    try:
        resp = session.post(f"{base_url}/v1/user-service/user/login", json=login_payload, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[ERROR] Failed to reach login endpoint: {e}")
        return 1

    token = data.get("accessToken")
    if token or data.get("success"):
        pass
    elif data.get("loginType") == "verifyCode":
        print("[INFO] 2FA verification required.")
        # Trigger sending verification code
        send_resp = session.post(f"{base_url}/v1/user-service/user/sendemail", json={"email": username, "type": "login"}, headers=headers)
        code = input("Please enter the verification code: ").strip()
        verify_payload = {
            "account": username,
            "code": code,
        }
        v_resp = session.post(f"{base_url}/v1/user-service/user/login", json=verify_payload, headers=headers)
        v_data = v_resp.json()
        token = v_data.get("accessToken")
        if not (token or v_data.get("success")):
            print(f"[ERROR] Verification failed: {v_data}")
            return 1
    else:
        print(f"[ERROR] Login failed: {data.get('message', data)}")
        return 1

    print("[SUCCESS] Login successful!")
    auth_headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": "BambuStudio/01.09.03.50",
    }

    print("\n[INFO] Fetching bound devices...")
    try:
        bind_resp = session.get(f"{base_url}/v1/iot-service/api/user/bind", headers=auth_headers, timeout=15)
        bind_resp.raise_for_status()
        bind_data = bind_resp.json()
    except Exception as e:
        print(f"[ERROR] Failed to fetch device list: {e}")
        return 1

    devices = bind_data.get("devices", [])
    if not devices:
        print("[WARN] No printers bound to this account.")
        return 0

    print(f"\nFound {len(devices)} device(s):\n" + "-" * 60)
    for idx, dev in enumerate(devices, 1):
        dev_id = dev.get("dev_id", "Unknown")
        name = dev.get("name", "Unnamed")
        model = dev.get("dev_product_name") or dev.get("dev_model_name", "Unknown")
        access_code = dev.get("dev_access_code", "N/A")
        online = dev.get("online", False)

        is_target_a1 = dev_id.startswith("039")
        marker = " [TARGET A1 PRINTER]" if is_target_a1 else ""

        print(f"Device #{idx}: {name} ({model}){marker}")
        print(f"  - Serial Number (dev_id) : {dev_id}")
        print(f"  - LAN Access Code        : {access_code}")
        print(f"  - Online Status          : {'Online' if online else 'Offline'}")
        print("-" * 60)

    return 0

if __name__ == "__main__":
    sys.exit(main())
