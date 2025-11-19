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
        "/api/robot/register",
        headers={
            "Authorization": token
        },
        json={
            "robot_id": "robot"
        }
    )

def test_create_cough_log():
    setup()

    response = client.post(
        "/api/cough/",
        json={
            "robot_id": "robot"
        }
    )

    assert response.status_code == 200


def test_create_cough_log():
    setup()

    response = client.post(
        "/api/cough/",
        json={
            "robot_id": "robot"
        }
    )

    assert response.status_code == 200