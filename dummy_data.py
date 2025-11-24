import os
import random
from datetime import datetime, timedelta, timezone

import httpx
from httpx import ConnectError

# Basic script to seed /api/retrain/sleep and /api/retrain/stress with dummy data.

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
EMAIL = os.getenv("DUMMY_EMAIL", "user@example.com")
PASSWORD = os.getenv("DUMMY_PASSWORD", "asdf1234")


def get_auth_headers(email: str = EMAIL, password: str = PASSWORD) -> dict:
    """Sign up (idempotent) and return Authorization header."""
    try:
        with httpx.Client(base_url=API_BASE_URL, timeout=10.0) as client:
            # Sign-up is idempotent for an existing user; ignore failure if already registered.
            client.post(
                "/api/auth/signup",
                json={"email": email, "password": password, "password2": password},
            )
            login = client.post(
                "/api/auth/login",
                json={"email": email, "password": password},
            )
            login.raise_for_status()
            body = login.json()
            token = f"{body['token_type']} {body['access_token']}"
            return {"Authorization": token}
    except ConnectError as exc:
        raise SystemExit(
            f"Cannot reach API at {API_BASE_URL}. "
            "Start the FastAPI server or set API_BASE_URL to the correct address."
        ) from exc


def random_sleep_payload(sleep_start_time: datetime, idx: int) -> dict:
    timestamp_ms = min(36000, idx * 1000 + random.randint(0, 1000))  # 0 ~ 36s offset
    return {
        "sleep_start_time": sleep_start_time.isoformat(),
        "timestamp": timestamp_ms,
        "acc_x": round(random.uniform(-2.0, 2.0), 3),
        "acc_y": round(random.uniform(-2.0, 2.0), 3),
        "acc_z": round(random.uniform(-2.0, 2.0), 3),
        "hr": round(random.uniform(55, 110), 2),
        "sleep_stage": random.choice([0, 1, 2, 3, 4]),
        "sao2": round(random.uniform(92, 100), 2),
        "bvp": round(random.uniform(0.0, 5.0), 3),
    }


def random_stress_payload() -> dict:
    return {
        "heart_rate_bpm": round(random.uniform(55, 130), 2),
        "hrv_sdnn_ms": round(random.uniform(20, 120), 2),
        "hrv_rmssd_ms": round(random.uniform(15, 110), 2),
        "acc_x_mean": round(random.uniform(-1.5, 1.5), 3),
        "acc_y_mean": round(random.uniform(-1.5, 1.5), 3),
        "acc_z_mean": round(random.uniform(-1.5, 1.5), 3),
        "acc_mag_mean": round(random.uniform(0.0, 3.0), 3),
        "acc_mag_std": round(random.uniform(0.0, 1.0), 3),
        "wrist_temperature_c_mean": round(random.uniform(31.0, 36.5), 2),
    }


def send_sleep_logs(headers: dict, count: int = 10):
    sleep_start = datetime.now(timezone.utc) - timedelta(hours=1)
    with httpx.Client(base_url=API_BASE_URL, headers=headers, timeout=10.0) as client:
        for idx in range(count):
            payload = random_sleep_payload(sleep_start, idx)
            resp = client.post("/api/retrain/sleep", json=payload)
            resp.raise_for_status()
            print(f"[sleep] sent #{idx+1}: {resp.json()}")


def send_stress_logs(headers: dict, count: int = 1000):
    with httpx.Client(base_url=API_BASE_URL, headers=headers, timeout=10.0) as client:
        for idx in range(count):
            payload = random_stress_payload()
            resp = client.post("/api/retrain/stress", json=payload)
            resp.raise_for_status()
            print(f"[stress] sent #{idx+1}: {resp.json()}")


def main():
    headers = get_auth_headers()
    send_sleep_logs(headers)
    send_stress_logs(headers)


if __name__ == "__main__":
    main()
