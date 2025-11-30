import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

DEVICE_ID = "sexydevice"

def get_token() -> str:
    _ = client.post(
        "/api/auth/signup",
        json={
            "email": "hi@naver.com",
            "password": "123",
            "password2": "123",
        },
    )
    login = client.post(
        "api/auth/login",
        json={
            "email": "hi@naver.com",
            "password": "123"
        }
    )
    body = login.json()
    token = "{} {}".format(body["token_type"], body["access_token"])
    return token


def test_create_device():
    token = get_token()

    response =  client.post(
        "api/device/register",
        headers={
            "Authorization": token
        },
        json={
            "device_id": DEVICE_ID
        }
    )

    assert response.status_code == 200



def test_delete_device():
    token = get_token()

    response =  client.delete(
        f"api/device/{DEVICE_ID}",
        headers={
            "Authorization": token
        }
    )

    assert response.status_code == 204


