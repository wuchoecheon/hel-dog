import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def get_token(email: str):
    _ = client.post(
        "/api/auth/signup",
        json={
            "email": email,
            "password": "asdf1234",
            "password2": "asdf1234",
        }
    )

    login = client.post(
        "/api/auth/login",
        json={
            "email": email,
            "password": "asdf1234",
        }
    )

    body = login.json()
    token = "{} {}".format(body["token_type"], body["access_token"])

    return token

def setup(device_id: str, email: str):
    token = get_token(email)
    response = client.post(
        "/api/device/register",
        headers={
            "Authorization": token
        },
        json={
            "device_id": device_id
        }
    )

    return token

def test_stress_log():
    device_id = "device_stress1"
    email = "user1@example.com"
    setup(device_id, email)

    response = client.post(
        f"/api/stress/{device_id}",
    )

    assert response.status_code == 200

def test_get_stress_status_false():
    device_id = "device_stress2"
    email = "user2@example.com"
    token = setup(device_id, email)

    client.post(
        f"/api/stress/{device_id}",
    )

    response = client.get(
        "/api/stress",
        headers = {"Authorization": token}
    )

    body = response.json()

    assert response.status_code == 200
    assert body["stressed"] is False
    assert body["count_last_12h"] == 1

def test_get_stress_status_true():
    device_id = "device_stress3"
    email = "user3@example.com"
    token = setup(device_id, email)

    client.post(
        f"/api/stress/{device_id}",
    )
    client.post(
        f"/api/stress/{device_id}",
    )
    client.post(
        f"/api/stress/{device_id}",
    )

    response = client.get(
        "/api/stress",
        headers = {"Authorization": token}
    )

    body = response.json()

    assert response.status_code == 200
    assert body["stressed"] is True
    assert body["count_last_12h"] == 3
