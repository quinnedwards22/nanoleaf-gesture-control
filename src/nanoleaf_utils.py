# nanoleaf_utils.py
import requests

# 1. Nanoleaf setup
NANO_IP    = "192.168.2.34"
AUTH_TOKEN = "9N4HGFpkLsa1jkBfbWxTdUk5FAzETGyP"
API_BASE   = f"http://{NANO_IP}:16021/api/v1/{AUTH_TOKEN}/state"

def set_brightness(bri: int):
    """Clamp to 0–100 and send to Nanoleaf."""
    bri = max(0, min(100, bri))
    payload = {"brightness": {"value": bri}}
    resp = requests.put(API_BASE, json=payload)
    resp.raise_for_status()

def get_brightness():
    """Get current brightness from Nanoleaf."""
    resp = requests.get(API_BASE)
    resp.raise_for_status()
    return resp.json()["brightness"]["value"]

def set_hue(hue: int):
    """Clamp to 0–360 and send hue (degrees) to Nanoleaf."""
    hue = hue % 360
    payload = {"hue": {"value": hue}}
    resp = requests.put(API_BASE, json=payload)
    resp.raise_for_status()

def set_power(on: bool):
    """Turn lights on/off."""
    payload = {"on": {"value": on}}
    resp = requests.put(API_BASE, json=payload)
    resp.raise_for_status()
