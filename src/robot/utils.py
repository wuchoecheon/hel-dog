from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

import jwt
from jwt.exceptions import InvalidTokenError

from fastapi.security import OAuth2PasswordBearer
from fastapi import Depends, HTTPException, status

import src.config as config
from src.robot.models import Robot
from src.auth.models import User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")

SECRET_KEY = config.SECRET_KEY
ALGORITHM = "HS256"

def get_current_user_email(token: str = Depends(oauth2_scheme)) -> str:
    """
    Authorization 헤더의 Bearer JWT 토큰에서 payload['sub'](이메일) 반환
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        return email
    except InvalidTokenError:
        raise credentials_exception
