import pytest
from fastapi.testclient import TestClient
from src.main import app

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

def setup():
    token = get_token()
    response = client.post(
        "/api/device/register",
        headers={
            "Authorization": token
        },
        json={
            "device_id": "device"
        }
    )

def test_stress_log():
    setup()

    response = client.post(
        "/api/stress/device",
    )

    assert response.status_code == 200

def test_get_stress_status_false():
    setup()

    client.post(
        "/api/stress/device",
    )

    token = get_token()

    response = client.get(
        "/api/stress",
        headers = {"Authorization": token}
    )

    body = response.json()

    assert response.status_code == 200
    assert body["stressed"] is False
    assert body["count_last_12h"] == 2

def test_get_stress_status_true():
    setup()

    client.post(
        "/api/stress/device",
    )

    token = get_token()

    response = client.get(
        "/api/stress",
        headers = {"Authorization": token}
    )

    body = response.json()

    assert response.status_code == 200
    assert body["stressed"] is True
    assert body["count_last_12h"] == 3
