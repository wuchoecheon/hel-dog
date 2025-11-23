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
            "robot_id": "robot"
        }
    )

def test_posture_log():
    setup()

    response = client.post(
        "/api/posture/robot",
        json={
            "label": 1,
        }
    )

    assert response.status_code == 200
    assert response.json() == {"response": "posture log saved"}, response.json()

def test_get_posture_last_3h():
    setup()

    response = client.post(
        "/api/posture/robot",
        json={
            "label": 1,
        }
    )
    response = client.post(
        "/api/posture/robot",
        json={
            "label": 2,
        }
    )
    response = client.post(
        "/api/posture/robot",
        json={
            "label": 2,
        }
    )
    response = client.post(
        "/api/posture/robot",
        json={
            "label": 2,
        }
    )

    token = get_token()

    response = client.get(
        "/api/posture",
        headers = {"Authorization": token}
    )

    body = response.json()
    counts = body["counts_by_label"]

    assert response.status_code == 200
    assert body["total"] == 5
    assert counts["1"] == 2
    assert counts["2"] == 3
    assert counts["3"] == 0
