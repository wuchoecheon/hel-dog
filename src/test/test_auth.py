import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_create_user():
    response = client.post(
        "/api/auth/signup",
        json={
            "email": "user1@example.com",
            "password": "asdf1234",
            "password2": "asdf1234",
        },
    )

    if(response.json() == {"response": "Email already registered"}):
        pass
    else:
        assert response.status_code == 200
        assert response.json() == {"response": "signup success!"}, response.json()



def test_create_existing_user():
    _ = client.post(
        "/api/auth/signup",
        json={
            "email": "user@example.com",
            "password": "asdf1234",
            "password2": "asdf1234",
        },
    )

    response = client.post(
        "/api/auth/signup",
        json={
            "email": "user@example.com",
            "password": "asdf1234",
            "password2": "asdf1234",
        },
    )
    assert response.status_code == 400



def test_signup_wrong_password():
    response = client.post(
        "/api/auth/signup",
        json={
            "email": "user@example.com",
            "password": "asdf1234",
            "password2": "qwer1234",
        },
    )
    assert response.status_code == 422


def test_login():
    response = client.post(
        "api/auth/login",
        json={
            "email": "user@example.com",
            "password": "asdf1234"
        }
    )
    
    assert response.status_code == 200


def test_login_with_invalid_password():
    response = client.post(
        "api/auth/login",
        json={
            "email": "user@example.com",
            "password": "qwer1234"
        }
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid email or password"}


def test_login_user_not_exists():
    response = client.post(
        "api/auth/login",
        json={
            "email": "invaliduser@example.com",
            "password": "asdf1234"
        }
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid email or password"}


def test_jwt_token():
    login = client.post(
        "api/auth/login",
        json={
            "email": "user@example.com",
            "password": "asdf1234"
        }
    )

    body = login.json()
    token = "{} {}".format(body["token_type"], body["access_token"])

    response =  client.post(
        "api/auth/check",
        headers={
            "Authorization": token
        }
    )

    assert response.status_code == 200