import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

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

def test_register_robot():
    headers = get_token()

    response = client.post(
        "/api/robot/register",
        json={
            "robot_id": "r1",
        },
        headers = headers,
    )

    if(response.json() == {"response": "Robot already registered"}):
        pass
    else:
        assert response.status_code == 200
        assert response.json() == {"response": "register success!"}, response.json()

def test_register_existing_robot():
    headers = get_token()

    _ = client.post(
        "/api/robot/register",
        json={
            "robot_id": "r2",
        },
        headers = headers,
    )

    response = client.post(
        "/api/robot/register",
        json={
            "robot_id": "r2",
        },
        headers = headers,
    )

    assert response.status_code == 400

def test_delete_robot():
    headers = get_token()

    register = client.post(
        "/api/robot/register",
        json={"robot_id": "r3"},
        headers=headers,
    )
    assert register.status_code == 200

    delete = client.delete(
        "/api/robot/r3",
        headers=headers,
    )

    assert delete.status_code == 204


def test_delete_unexisting_robot():
    headers = get_token()

    delete = client.delete(
        "/api/robot/invalidRobot",
        headers=headers,
    )

    assert delete.status_code == 404

def test_delete_other_user_robot():
    headers_reg = get_token()
    headers_del = get_token(email="user2@example.com")

    register = client.post(
        "/api/robot/register",
        json={"robot_id": "r4"},
        headers=headers_reg,
    )
    assert register.status_code == 200

    delete = client.delete(
        "/api/robot/r4",
        headers=headers_del,
    )

    assert delete.status_code == 400
