import pytest
from fastapi.testclient import TestClient
from src.main import app
from datetime import datetime, timedelta

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

    headers = {
        "Authorization": token
    }

    return headers

def register_robot(robot_id="r"):
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
    register_robot(robot_id)

    response = client.post(
        "/api/fall/log",
        json={
            "robot_id": robot_id,
            "label": "fall",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"response": "fall log saved"}, response.json()

def test_fall_log_invalid_robot():
    response = client.post(
        "/api/fall/log",
        json={
            "robot_id": "invalidRobot",
            "label": "fall",
        },
    )

    assert response.status_code == 400

def test_get_fall_today():
    headers = get_token()

    robot_id = "fall2"
    register_robot(robot_id)

    # 오늘 fall log
    _ = client.post(
        "/api/fall/log",
        json={
            "robot_id": robot_id,
            "label": "fall",
        },
    )
    _ = client.post(
        "/api/fall/log",
        json={
            "robot_id": robot_id,
            "label": "fall",
        },
    )
    # 오늘 normal log
    _ = client.post(
        "/api/fall/log",
        json={
            "robot_id": robot_id,
            "label": "normal",
        },
    )
    # 어제 fall log
    _ = client.post(
        "/api/fall/log",
        json={
            "robot_id": robot_id,
            "label": "fall",
            "occurred_at": (datetime.now() - timedelta(days=1)).isoformat(),
        }
    )

    response = client.get(
        "/api/fall",
        headers = headers,
    )
    assert response.status_code == 200

    data = response.json()
    logs = [item for item in data if item["robot_id"] == robot_id]

    assert len(logs) == 2
    assert all(item["label"] == "fall" for item in logs)
