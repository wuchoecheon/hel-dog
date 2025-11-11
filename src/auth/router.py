from fastapi import APIRouter, Depends, HTTPException, status
from typing import Annotated
from sqlalchemy.orm import Session

from src.database import get_db
from src.auth.models import User
from src.auth.schemas import SignupSchema, LoginSchema, Token
from src.auth.utils import hashpw, checkpw, create_access_token, get_user

router = APIRouter(
    prefix="/api/auth"
)

@router.post("/signup")
def auth_signup(user: SignupSchema, db: Session=Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered"
        )

    hashed_password = hashpw(user.password)
    db_user = User(
        email=user.email, password=hashed_password
    )

    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    return {"response": "signup success!"}


@router.post("/login")
def auth_login(user: LoginSchema, db: Annotated[Session, Depends(get_db)]):
    db_user = db.query(User).filter(User.email == user.email).first()
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid email or password"
        )

    if not checkpw(user.password, db_user.password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid email or password"
        )

    access_token = create_access_token(
        data={"sub": user.email}
    )
    return Token(access_token=access_token, token_type="bearer")




@router.post("/check")
def validate_token(user: Annotated[User, Depends(get_user)]):
    return {"email": user.email}