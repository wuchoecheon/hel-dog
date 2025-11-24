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

def setup(robot_id: str = "robot", device_id: str = "device"):
    token = get_token()

    _ = client.post(
        "/api/robot/register",
        headers={"Authorization": token},
        json={
            "robot_id": robot_id,
        }
    )

    _ = client.post(
        "/api/device/register",
        headers={"Authorization": token},
        json={
            "device_id": device_id,
        }
    )

    return token

def test_create_healthdata():
    device_id = "device_health1"
    setup(device_id=device_id)

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

def test_get_health_score():
    robot_id = "robot_health2"
    device_id = "device_health2"
    token = setup(robot_id, device_id)

    client.post(f"/api/cough/{robot_id}")
    client.post(f"/api/cough/{robot_id}")

    client.post(
        f"/api/stress/{device_id}",
        headers={"Authorization": token},
    )

    client.post(
        f"/api/sleep/{device_id}",
        headers={"Authorization": token},
        json={"label": 3},
    )

    res = client.get(
        "/api/health/score",
        headers={"Authorization": token},
    )

    assert res.status_code == 200
    body = res.json()

    components = body["components"]
    for key in [
        "sleep_score",
        "cough_score",
        "cough_count_3h",
        "stress_score",
        "stress_count_12h",
    ]:
        assert key in components

    assert body["status"] in ["good", "moderate", "bad"]
