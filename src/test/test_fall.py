import pytest
from fastapi.testclient import TestClient
from src.main import app
from datetime import datetime, timedelta

client = TestClient(app)

def get_token(email="user@example.com"):
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

    headers = {
        "Authorization": token
    }

    return headers

def register_robot(robot_id="r", headers=None):
    if headers is None:
        headers = get_token()

    register = client.post(
        "/api/robot/register",
        json={
            "robot_id": robot_id,
        },
        headers = headers,
    )

    return register

def test_fall_log():
    robot_id = "fall1"
    headers = get_token()
    register_robot(robot_id, headers=headers)

    response = client.post(
        f"/api/fall/{robot_id}",
    )

    assert response.status_code == 200
    assert response.json() == {"response": "fall log saved"}, response.json()

def test_fall_log_invalid_robot():
    response = client.post(
        "/api/fall/invalidRobot",
    )

    assert response.status_code == 400

def test_get_fall_today():
    headers = get_token(email="user2@example.com")

    robot_id = "fall2"
    register_robot(robot_id, headers=headers)

    # 오늘 fall log
    _ = client.post(
        f"/api/fall/{robot_id}",
    )
    _ = client.post(
        f"/api/fall/{robot_id}",
    )

    response = client.get(
        "/api/fall",
        headers = headers,
    )
    assert response.status_code == 200

    data = response.json()

    assert data["fall_num"] == 2
    assert len(data["fall_log"]) == 2
