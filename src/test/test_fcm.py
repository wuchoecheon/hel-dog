import pytest
from fastapi.testclient import TestClient
from src.main import app

from src.fcm.utils import send_notification

client = TestClient(app)

uuid_sample = "c8ebf2fd-8284-4070-9f3e-82c904f43de7"
fcm_token_sample="fill_me"

def get_token(email="user@example.com", password="asdf1234"):
    _ = client.post(
        "/api/auth/signup",
        json={
            "email": email,
            "password": password,
            "password2": password
        }
    )
    login = client.post(
        "/api/auth/login",
        json={
            "email": email,
            "password": password
        }
    )

    body = login.json()
    token = "{} {}".format(body["token_type"], body["access_token"])

    headers = {
        "Authorization": token
    }

    return headers

def test_fcm_register():
    headers = get_token()


    response = client.post(
        "/api/fcm/register",
        json={
            "uuid": uuid_sample,
            "fcm_token": "thisisfcmtoken",
        },
        headers = headers,
    )

    assert response.status_code == 200


def test_fcm_sendnotification():
    print(send_notification(fcm_token=fcm_token_sample, title="sample title", body="this is body"))
    assert True == False
