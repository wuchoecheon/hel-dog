# tests/test_sleep.py
import pytest
from fastapi.testclient import TestClient
from src.main import app

from datetime import datetime

client = TestClient(app)

def get_token():
    _ = client.post(
        "/api/auth/signup",
        json={
            "email": "user@example.com",
            "password": "asdf1234",
            "password2": "asdf1234",
        }
    )

    login = client.post(
        "/api/auth/login",
        json={
            "email": "user@example.com",
            "password": "asdf1234",
        }
    )

    body = login.json()
    token = "{} {}".format(body["token_type"], body["access_token"])
    return token


def setup_user_and_device(device_id: str = "device"):
    token = get_token()

    _ = client.post(
        "/api/device/register",
        headers={"Authorization": token},
        json={
            "device_id": device_id,
        }
    )


def test_create_healthdata():
    device_id = "device"
    setup_user_and_device(device_id)

    response = client.post(
        f"/api/health/{device_id}",
        json={ 
            "timestamp": str(datetime.now()),
            "heartRateData": 60,
            "oxygenSaturation": 0.012,
            "stressLevel": "low",
        }
    )

    assert response.status_code == 200
