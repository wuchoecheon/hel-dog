from fastapi import APIRouter, Depends, HTTPException, status
from typing import Annotated
from sqlalchemy.orm import Session

from src.database import get_db
from src.auth.models import User
from src.auth.utils import get_user
from src.fcm.models import FCM
from src.fcm.schemas import FCMRegisterSchema

router = APIRouter(
    prefix="/api/fcm"
)

@router.post("/register")
def fcm_register(body: FCMRegisterSchema, user: Annotated[User, Depends(get_user)], db: Session=Depends(get_db)):
    
    db_fcm = FCM(
        uuid=body.uuid,
        fcm_token=body.fcm_token,
        owner=user.email
    )
    
    db.add(db_fcm)
    db.commit()
    db.refresh(db_fcm)

    return {"response": "register success!"}