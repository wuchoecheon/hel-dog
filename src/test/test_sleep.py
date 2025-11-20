# tests/test_sleep.py
import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def get_token():
    # 이미 가입되어 있어도 상관없이 그냥 호출
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

    # cough 쪽의 /api/robot/register 에 대응하는 device 등록 API가 있다고 가정
    # 경로/필드명이 다르면 여기만 맞춰서 수정하면 됨.
    _ = client.post(
        "/api/device/register",
        headers={"Authorization": token},
        json={
            "device_id": device_id,
        }
    )


def test_create_sleep_log():
    device_id = "device"
    setup_user_and_device(device_id)

    response = client.post(
        f"/api/sleep/{device_id}",
        json={
            "label": 1,   # SleepLogSchema 에 최소 label 필드는 있다고 가정
        },
    )

    assert response.status_code == 200


def test_get_sleep_score():
    device_id = "device"
    setup_user_and_device(device_id)

    NUM = 5

    for i in [1,2,3,4,3,2,1]:
        for _ in range(NUM):
            client.post(
                f"/api/sleep/{device_id}",
                json={"label": i},
            )

    token = get_token()

    response = client.get(
        "/api/sleep",
        headers={"Authorization": token},
    )

    body = response.json()

    assert response.status_code == 200
    assert body["response"] == "request proceed successfully"
    assert body["sleep_score"] == 86
